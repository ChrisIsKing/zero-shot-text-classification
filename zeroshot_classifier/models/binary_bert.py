import math
import pickle
import random
from typing import List, Dict
from os.path import join
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util.load_data import (
    get_datasets, binary_cls_format, in_domain_data_path, out_of_domain_data_path
)
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.models.architecture import load_sliced_binary_bert, BinaryBertCrossEncoder


MODEL_NAME = 'Binary BERT'
HF_MODEL_NAME = 'bert-base-uncased'


def parse_args():
    modes = sconfig('training.strategies')

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--max_sequence_length', type=int, default=512)
    parser_train.add_argument('--output', type=str, default=None)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], default='rand')
    # model to initialize weights from, intended for loading weights from local explicit training
    parser_train.add_argument('--model_init', type=str, default=HF_MODEL_NAME)
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--learning_rate', type=float, default=2e-5)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    # set test arguments
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_test.add_argument('--batch_size', type=int, default=32)  # #of texts to do inference in a single forward pass
    parser_test.add_argument('--model_path', type=str, required=True)
    
    return parser.parse_args()


if __name__ == '__main__':
    import os

    import numpy as np
    import transformers

    seed = sconfig('random-seed')

    # INTENT_ONLY = True
    INTENT_ONLY = False
    # NORMALIZE_ASPECT = False
    NORMALIZE_ASPECT = True
    if INTENT_ONLY:
        def filt(d, dom):
            return d['domain'] == dom and d['aspect'] == 'intent'
    else:
        def filt(d, dom):
            return d['domain'] == dom

    args = parse_args()
    cmd = args.command
    logger = get_logger(f'{MODEL_NAME} {args.command.capitalize()}')

    if cmd == 'train':
        output_path, sampling, mode = args.output, args.sampling, args.mode
        lr, bsz, n_ep = args.learning_rate, args.batch_size, args.epochs
        model_init, seq_len = args.model_init, args.max_sequence_length

        # n = None
        n = 64

        dset_args = dict(normalize_aspect=seed) if NORMALIZE_ASPECT else dict()
        data = get_datasets(domain='in', n_sample=n, **dset_args)
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if filt(d_dset, 'in')]
        logger.info(f'Processing datasets {pl.i(dataset_names)} for training... ')
        train = []
        val = []
        test = []
        for dataset_name in dataset_names:
            dset = data[dataset_name]
            args = dict(dataset_name=dataset_name, sampling=sampling, mode=mode)
            train += binary_cls_format(dset, **args, split='train')
            val += binary_cls_format(dset, **args, split='eval')
            test += binary_cls_format(dset, **args, split='test')

        # in case of loading from explicit pre-training,
        # the classification head would be ignored for classifying 3 classes
        d_log = dict(model_init=model_init)
        if model_init != HF_MODEL_NAME:
            d_log['files'] = os.listdir(model_init)
        logger.info(f'Loading model with {pl.i(d_log)}...')
        model = BinaryBertCrossEncoder(model_init, num_labels=2, automodel_args=dict(ignore_mismatched_sizes=True))
        if seq_len != 512:  # Intended for `bert-base-uncased` only; TODO: binary bert seems to support this already?
            model.tokenizer, model.model = load_sliced_binary_bert(model_init, seq_len)

        spec_tok_arg = utcd_util.get_add_special_tokens_args(model.tokenizer, train_strategy=mode)
        if spec_tok_arg:
            logger.info(f'Adding special tokens {pl.i(spec_tok_arg)} to tokenizer... ')
            model.tokenizer.add_special_tokens(special_tokens_dict=spec_tok_arg)
            model.model.resize_token_embeddings(len(model.tokenizer))

        transformers.logging.set_verbosity_error()  # disables `longest_first` warning
        random.seed(seed)
        random.shuffle(train)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=bsz)
        val_dataloader = DataLoader(val, shuffle=False, batch_size=bsz)

        warmup_steps = math.ceil(len(train_dataloader) * n_ep * 0.1)  # 10% of train data for warm-up
        d_log = {'#data': len(train), 'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps}
        logger.info(f'Launched training with {pl.i(d_log)}... ')

        output_path = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), output_path=output_path,
            mode=mode, sampling=sampling, normalize_aspect=NORMALIZE_ASPECT
        )
        logger.info(f'Model will be saved to {pl.i(output_path)}')

        transformers.set_seed(seed)
        model.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=n_ep,
            optimizer_params=dict(lr=lr),
            warmup_steps=warmup_steps,
            output_path=output_path
        )
    elif cmd == 'test':
        WITH_EVAL_LOSS = False
        mode, domain, model_path, bsz = args.mode, args.domain, args.model_path, args.batch_size
        split = 'test'

        out_path = join(model_path, 'eval', domain2eval_dir_nm(domain))
        os.makedirs(out_path, exist_ok=True)

        data = get_datasets(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        model = CrossEncoder(model_path)  # load model

        logger = get_logger(f'{MODEL_NAME} Eval')
        d_log = dict(mode=mode, domain=domain, batch_size=bsz, path=model_path)
        logger.info(f'Evaluating Binary Bert with {pl.i(d_log)} and saving to {pl.i(out_path)}... ')

        eval_loss: Dict[str, np.array] = dict()  # a sense of how badly the model makes the prediction
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if filt(d_dset, domain)]

        for dnm in dataset_names:  # loop through all datasets
            dset = data[dnm]
            pairs, aspect = dset[split], dset['aspect']
            d_dset = sconfig(f'UTCD.datasets.{dnm}.splits.{split}')
            label_options, multi_label = d_dset['labels'], d_dset['multi_label']
            n_options = len(label_options)
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            d_log = {'#text': n_txt, '#label': n_options, 'labels': label_options}
            logger.info(f'Evaluating {pl.i(dnm)} with {pl.i(d_log)}...')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            arr_loss = torch.empty(n_txt, dtype=torch.float32) if WITH_EVAL_LOSS else None

            txt_n_lbs2query = TrainStrategy2PairMap(train_strategy=mode)(aspect)

            gen = group_n(pairs.items(), n=bsz)
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
            df, acc = eval_res2df(arr_labels, arr_preds, report_args=args)
            logger.info(f'{pl.i(dnm)} Classification Accuracy: {pl.i(acc)}')
            df.to_csv(join(out_path, f'{dnm}.csv'))

        if WITH_EVAL_LOSS:
            with open(join(out_path, 'eval_loss.pkl'), 'wb') as f:
                pickle.dump(eval_loss, f)
