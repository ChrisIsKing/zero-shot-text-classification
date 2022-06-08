import math
import pickle
import random
import logging
import datetime
from typing import List, Dict
from os.path import join
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util.load_data import get_data, binary_cls_format, in_domain_data_path, out_of_domain_data_path
from zeroshot_classifier.baseline.architecture import load_sliced_binary_bert


def parse_args():
    modes = [
        'vanilla',
        'implicit',
        'implicit-on-text-encode-aspect',  # encode each of the 3 aspects as 3 special tokens, followed by text
        'implicit-on-text-encode-sep',  # encode aspects normally, but add special token between aspect and text
        'explicit'  # see `zeroshot_classifier.explicit.binary_bert.py` for explicit training
    ]

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--max_sequence_length', type=int, default=512)
    parser_train.add_argument('--output', type=str, required=True)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], required=True)
    # model to initialize weights from, intended for loading weights from local explicit training
    parser_train.add_argument('--model_init', type=str, default='bert-base-uncased')
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    # set test arguments
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_test.add_argument('--batch_size', type=int, default=32)  # #of texts to do inference in a single forward pass
    parser_test.add_argument('--model_path', type=str, required=True)
    
    return parser.parse_args()


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    import os

    import numpy as np
    import transformers

    from icecream import ic

    seed = sconfig('random-seed')

    # INTENT_ONLY = True
    INTENT_ONLY = False
    NORMALIZE_ASPECT = False
    # NORMALIZE_ASPECT = True
    if INTENT_ONLY:
        def filt(d, dom):
            return d['domain'] == dom and d['aspect'] == 'intent'
    else:
        def filt(d, dom):
            return d['domain'] == dom

    args = parse_args()
    if args.command == 'train':
        model_init, seq_len = args.model_init, args.max_sequence_length
        dset_args = dict(normalize_aspect=seed) if NORMALIZE_ASPECT else dict()
        data = get_data(in_domain_data_path, **dset_args)
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if filt(d_dset, 'in')]
        ic(dataset_names)
        train = []
        test = []
        for dataset_name in dataset_names:
            dset = data[dataset_name]
            train += binary_cls_format(dset, name=dataset_name, sampling=args.sampling, mode=args.mode)
            test += binary_cls_format(dset, train=False, mode=args.mode)

        train_batch_size = args.batch_size
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        # in case of loading from explicit pre-training,
        # the classification head would be ignored for classifying 3 classes
        model = CrossEncoder(model_init, num_labels=2, automodel_args=dict(ignore_mismatched_sizes=True))
        if seq_len != 512:  # Intended for `bert-base-uncased` only
            model.tokenizer, model.model = load_sliced_binary_bert(model_init, seq_len)
        spec_tok_args = dict(eos_token='[eot]')  # Add end of turn token for sgd
        add_spec_toks = None
        if args.mode == 'implicit-on-text-encode-aspect':
            add_spec_toks = list(sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token').values())
        elif args.mode == 'implicit-on-text-encode-sep':
            add_spec_toks = [sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')]
        if add_spec_toks:
            spec_tok_args.update(dict(additional_special_tokens=add_spec_toks))
        model.tokenizer.add_special_tokens(spec_tok_args)
        model.model.resize_token_embeddings(len(model.tokenizer))

        transformers.logging.set_verbosity_error()  # disables `longest_first` warning
        random.seed(seed)
        new_shuffle = True
        random.shuffle(train)  # TODO: always need this?
        if new_shuffle:
            train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)
        else:
            train_dataloader = DataLoader(train, shuffle=False, batch_size=train_batch_size)
        ic(new_shuffle)

        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test, name='UTCD-test')

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

        transformers.set_seed(seed)
        # Train the model
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=100000,
            warmup_steps=warmup_steps,
            output_path=model_save_path
        )
    if args.command == 'test':
        WITH_EVAL_LOSS = False
        mode, domain, model_path, bsz = args.mode, args.domain, args.model_path, args.batch_size
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        date = datetime.datetime.now().strftime('%m.%d.%Y')
        date = date[:-4] + date[-2:]  # 2-digit year
        out_path = join(model_path, 'eval', f'{domain_str}, {date}')
        os.makedirs(out_path, exist_ok=True)

        data = get_data(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        model = CrossEncoder(model_path)  # load model
        sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')
        aspect2aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')

        logger = get_logger('Binary Bert Eval')
        d_log = dict(mode=mode, domain=domain, batch_size=bsz, path=model_path)
        logger.info(f'Evaluating Binary Bert with {log_dict(d_log)} and saving to {logi(out_path)}... ')

        eval_loss: Dict[str, np.array] = dict()  # a sense of how badly the model makes the prediction
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if filt(d_dset, domain)]

        for dnm in dataset_names:  # loop through all datasets
            dset = data[dnm]
            split = 'test'
            txts, aspect = dset[split], dset['aspect']
            d_dset = sconfig(f'UTCD.datasets.{dnm}.splits.{split}')
            label_options, multi_label = d_dset['labels'], d_dset['multi_label']
            n_options = len(label_options)
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            d_log = {'#text': n_txt, '#label': n_options}
            logger.info(f'Evaluating {logi(dnm)} with {log_dict(d_log)}...')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            arr_loss = torch.empty(n_txt, dtype=torch.float32) if WITH_EVAL_LOSS else None

            txt_n_lbs2query = None
            if mode in ['vanilla', 'explicit']:
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, lb] for lb in lbs]
            elif mode == 'implicit':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, f'{lb} {aspect}'] for lb in lbs]
            elif mode == 'implicit-on-text-encode-aspect':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect2aspect_token[aspect]} {txt}', lb] for lb in lbs]
            elif mode == 'implicit-on-text-encode-sep':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect} {sep_token} {txt}', lb] for lb in lbs]

            gen = group_n(txts.items(), n=bsz)
            # loop through each test example
            for i_grp, group in enumerate(tqdm(gen, desc=dnm, unit='group', total=math.ceil(n_txt/bsz))):
                txts_, lst_labels = zip(*group)
                lst_labels: List[List[int]] = [[label2id[lb] for lb in labels] for labels in lst_labels]
                query = sum([txt_n_lbs2query(t, label_options) for t in txts_], start=[])  # (n_options x bsz, 2)
                # probability for positive class
                logits = model.predict(query, batch_size=bsz, apply_softmax=True, convert_to_tensor=True)[:, 1]
                logits = logits.reshape(-1, n_options)
                preds = logits.argmax(axis=1)
                trues = torch.empty_like(preds)
                for i, pred, labels in zip(range(bsz), preds, lst_labels):
                    # if false prediction, pick one of the correct labels arbitrarily
                    trues[i] = pred if pred in labels else labels[0]
                idx_strt = i_grp*bsz
                arr_preds[idx_strt:idx_strt+bsz], arr_labels[idx_strt:idx_strt+bsz] = preds.cpu(), trues.cpu()
                if WITH_EVAL_LOSS:
                    if multi_label and any(len(lbs) > 1 for lbs in lst_labels):
                        # in this case, vectorizing is complicated, run on each sample separately since edge case anyway
                        for i, lbs in enumerate(lst_labels):
                            target = torch.tensor(lbs, device=logits.device)
                            if len(lbs) > 1:
                                loss = max(F.cross_entropy(logits[i].repeat(len(lbs), 1), target, reduction='none'))
                            else:
                                loss = F.cross_entropy(logits[None, i], target)  # dummy batch dimension
                            arr_loss[idx_strt+i] = loss
                    else:
                        arr_loss[idx_strt:idx_strt+bsz] = F.cross_entropy(logits, trues, reduction='none')
            if WITH_EVAL_LOSS:
                eval_loss[dnm] = arr_loss.numpy()

            args = dict(zero_division=0, target_names=label_options, output_dict=True)  # disables warning
            report = classification_report(arr_labels, arr_preds, **args)
            acc = f'{report["accuracy"]:.3f}'
            logger.info(f'{logi(dnm)} Classification Accuracy: {logi(acc)}')
            df = pd.DataFrame(report).transpose()
            df.to_csv(join(out_path, f'{dnm}.csv'))
        if WITH_EVAL_LOSS:
            with open(join(out_path, 'eval_loss.pkl'), 'wb') as f:
                pickle.dump(eval_loss, f)
