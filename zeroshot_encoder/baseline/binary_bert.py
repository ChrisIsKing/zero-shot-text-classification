import math
import random
import logging
import datetime
from typing import List
from os.path import join
from argparse import ArgumentParser

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from tqdm import tqdm

from stefutil import *
from zeroshot_encoder.util import *
from zeroshot_encoder.util.load_data import get_data, binary_cls_format, in_domain_data_path, out_of_domain_data_path


def parse_args():
    modes = [
        'vanilla',
        'implicit',
        'implicit-on-text-encode-aspect',  # encode each of the 3 aspects as 3 special tokens, followed by text
        'implicit-on-text-encode-sep',  # encode aspects normally, but add special token between aspect and text
        # see `zeroshot_encoder.explicit.binary_bert.py` for explicit training
    ]

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--output', type=str, required=True)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], required=True)
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    # set test arguments
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_test.add_argument('--batch_size', type=int, default=32)  # #of texts to do inference in a single forward pass
    # parser_test.add_argument('--group_size', type=int, default=16)
    parser_test.add_argument('--model_path', type=str, required=True)
    
    return parser.parse_args()


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import os

    import numpy as np
    import transformers

    from icecream import ic

    transformers.set_seed(42)

    args = parse_args()
    if args.command == 'train':
        data = get_data(in_domain_data_path)
        # get keys from data dict
        dataset_names = list(data.keys())
        train = []
        test = []
        for dataset in dataset_names:
            train += binary_cls_format(data[dataset], name=dataset, sampling=args.sampling, mode=args.mode)
            test += binary_cls_format(data[dataset], train=False, mode=args.mode)

        train_batch_size = args.batch_size
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        model = CrossEncoder('bert-base-uncased', num_labels=2)
        spec_tok_args = dict(eos_token='[eot]')  # Add end of turn token for sgd
        add_spec_toks = None
        if args.mode == 'implicit-on-text-encode-aspect':
            add_spec_toks = list(sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token').values())
        elif args.mode == 'implicit-on-text-encode-sep':
            add_spec_toks = [sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')]
        if add_spec_toks:
            spec_tok_args |= dict(additional_special_tokens=add_spec_toks)
        model.tokenizer.add_special_tokens(spec_tok_args)
        model.model.resize_token_embeddings(len(model.tokenizer))

        new_shuffle = True
        if new_shuffle:
            train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)
        else:
            random.shuffle(train)
            train_dataloader = DataLoader(train, shuffle=False, batch_size=train_batch_size)
        ic(new_shuffle)

        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test, name='UTCD-test')

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

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
        mode, domain, model_path, bsz = args.mode, args.domain, args.model_path, args.batch_size
        domain_str = 'in-domain' if domain == 'in' else 'out-domain'
        date = datetime.datetime.now().strftime('%Y%m.%d')
        out_path = join(model_path, 'eval', f'{domain_str}, {date}')
        os.makedirs(out_path, exist_ok=True)

        data = get_data(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        model = CrossEncoder(model_path)  # load model
        sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')
        aspect2aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')

        logger = get_logger('Binary Bert Eval')
        d_log = dict(mode=mode, domain=domain, batch_size=bsz, path=model_path)
        logger.info(f'Evaluating Binary Bert with {log_dict(d_log)}... ')

        # loop through all datasets
        for dnm, dset in data.items():
            if 'amazon' not in dnm:
                continue
            split = 'test'
            txts, aspect = dset[split], dset['aspect']
            label_options = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.labels')
            n_options = len(label_options)
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            logger.info(f'Evaluating {dnm} with {logi(n_txt)} texts...')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)

            txt_n_lbs2query = None
            if mode == 'vanilla':
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
            idxs_added = set()
            # ic(model._target_device)
            # loop through each test example
            for i_grp, group in enumerate(tqdm(gen, desc=dnm, unit='group', total=math.ceil(n_txt/bsz))):
                txts_, lst_labels = zip(*group)
                # ic(txts_, lst_labels)
                idxs = [i + i_grp*bsz for i in range(bsz)]
                # ic(5420 in idxs)
                query = sum([txt_n_lbs2query(t, label_options) for t in txts_], start=[])  # (n_options x bsz, 2)
                # ic(len(query))
                # probability for positive class
                logits = model.predict(query, apply_softmax=True, batch_size=bsz)[:, 1].reshape(-1, n_options)
                # ic(logits.shape)
                preds = logits.argmax(axis=1)
                # ic(preds)
                for i, pred, labels in zip(idxs, preds, lst_labels):
                    ids_labels = [label2id[lb] for lb in labels]
                    # if false prediction, pick one of the correct labels arbitrarily
                    true = pred if pred in ids_labels else ids_labels[0]
                    arr_preds[i], arr_labels[i] = pred, true
                    idxs_added.add(i)
                # exit(1)
            # idxs_not_added = sorted(set(range(n_txt)) - idxs_added)
            # ic(idxs_not_added)
            assert np.all((0 <= arr_preds) & (arr_preds < n_options))  # sanity check
            assert np.all((0 <= arr_labels) & (arr_labels < n_options))

            report = classification_report(
                arr_labels, arr_preds, target_names=label_options,
                output_dict=True
            )
            # ic(report)
            # ic(np.sum(arr_preds == arr_labels) / arr_preds.size)
            acc = f'{report["accuracy"]:.3f}'
            logger.info(f'{logi(dnm)} Classification Accuracy: {logi(acc)}')
            df = pd.DataFrame(report).transpose()
            df.to_csv(join(out_path, f'{dnm}.csv'))
            exit(1)
