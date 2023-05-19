import math
from os.path import join as os_join
from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
import pandas as pd
import torch.cuda
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.util.load_data import (
    get_datasets, seq_cls_format, in_domain_data_path, out_of_domain_data_path
)


MODEL_NAME = 'BERT Seq CLS'
HF_MODEL_NAME = 'bert-base-uncased'


def parse_args():
    parser = ArgumentParser()

    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    parser_train.add_argument('--dataset', type=str, default='all')
    parser_train.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_train.add_argument('--normalize_aspect', type=bool, default=False)
    parser_train.add_argument('--learning_rate', type=float, default=5e-5)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    parser_test.add_argument('--dataset', type=str, default='all')
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--model_name_or_path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    import os

    import transformers
    import datasets

    args = parse_args()

    seed = sconfig('random-seed')

    if args.command == 'train':
        logger = get_logger(f'{MODEL_NAME} Train')
        dataset_name, domain, normalize_aspect = args.dataset, args.domain, args.normalize_aspect
        lr, bsz, n_ep = args.learning_rate, args.batch_size, args.epochs
        ca(dataset_domain=domain)
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'

        dset_args = dict(domain=domain)
        if normalize_aspect:
            dset_args['normalize_aspect'] = seed
        data = get_datasets(**dset_args)
        if dataset_name == 'all':
            train_dset, test_dset, labels = seq_cls_format(data, all=True)
        else:
            train_dset, test_dset, labels = seq_cls_format(data[dataset_name])
        d_log = {'#train': len(train_dset), '#test': len(test_dset), 'labels': list(labels.keys())}
        logger.info(f'Loaded {pl.i(domain_str)} datasets {pl.i(dataset_name)} with {pl.i(d_log)} ')

        num_labels = len(labels)
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(HF_MODEL_NAME, return_dict=True, num_labels=num_labels)
        tokenizer.add_special_tokens(dict(eos_token=utcd_util.EOT_TOKEN))  # end-of-turn for SGD
        model.resize_token_embeddings(len(tokenizer))

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        train_dset = Dataset.from_pandas(pd.DataFrame(train_dset))
        test_dset = Dataset.from_pandas(pd.DataFrame(test_dset))
        # small batch size cos samples are very long in some datasets
        map_args = dict(batched=True, batch_size=16, num_proc=os.cpu_count())
        train_dset = train_dset.map(tokenize_function, **map_args)
        test_dset = test_dset.map(tokenize_function, **map_args)

        warmup_steps = math.ceil(len(train_dset) * n_ep * 0.1)  # 10% of train data for warm-up

        dir_nm = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), output_path=f'{domain}-{dataset_name}', mode=None,
            sampling=None, normalize_aspect=normalize_aspect
        )
        output_path = os_join(utcd_util.get_base_path(), u.proj_dir, u.model_dir, dir_nm)
        proj_output_path = os_join(u.base_path, u.proj_dir, u.model_dir, dir_nm, 'trained')
        d_log = {'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps, 'save path': output_path}
        logger.info(f'Launched training with {pl.i(d_log)}... ')

        training_args = TrainingArguments(  # TODO: learning rate
            output_dir=output_path,
            learning_rate=lr,
            num_train_epochs=n_ep,
            per_device_train_batch_size=bsz,
            per_device_eval_batch_size=bsz,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir='./logs',
            load_best_model_at_end=True,
            logging_steps=100000,
            save_steps=100000,
            evaluation_strategy='steps'
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dset, eval_dataset=test_dset, compute_metrics=compute_metrics
        )
        transformers.set_seed(seed)
        trainer.train()
        trainer.save_model(proj_output_path)
        tokenizer.save_pretrained(proj_output_path)
    elif args.command == 'test':
        dataset_name, domain, model_path = args.dataset, args.domain, args.model_name_or_path
        bsz = 32
        split = 'test'
        dataset_names = utcd_util.get_dataset_names(domain)
        if dataset_name != 'all':
            assert dataset_name in dataset_names
            dataset_names = [dataset_name]
        output_path = os_join(model_path, 'eval')
        lg_nm = f'{MODEL_NAME} Eval'
        logger = get_logger(lg_nm)
        lg_fl = os_join(output_path, f'{now(for_path=True)}_{lg_nm}, dom={domain}.log')
        logger_fl = get_logger(lg_nm, kind='file-write', file_path=lg_fl)
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        logger.info(f'Evaluating {pl.i(domain_str)} datasets {pl.i(dataset_names)} on model {pl.i(model_path)}... ')
        logger_fl.info(f'Evaluating {domain_str} datasets {dataset_names} on model {model_path}... ')

        data = get_datasets(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        device = 'cpu'
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'

        lb2id: Dict[str, int] = dict()  # see `load_data.seq_cls_format`
        if dataset_name == 'all':
            for dset in data.values():
                for label in dset['labels']:
                    if label not in lb2id:
                        lb2id[label] = len(lb2id)
        else:
            for label in data[dataset_name]['labels']:
                if label not in lb2id:
                    lb2id[label] = len(lb2id)
        _lbs = list(lb2id.keys())
        logger.info(f'Loaded labels: {pl.i(_lbs)}')
        logger_fl.info(f'Loaded labels: {_lbs}')

        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        for dnm in dataset_names:
            pairs: Dict[str, List[str]] = data[dnm][split]
            asp = sconfig(f'UTCD.datasets.{dnm}.aspect')
            logger.info(f'Evaluating {pl.i(asp)} dataset {pl.i(dnm)}... ')
            logger_fl.info(f'Evaluating {asp} dataset {dnm}... ')

            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            logger.info(f'Loading {pl.i(n_txt)} samples... ')
            logger_fl.info(f'Loading {n_txt} samples... ')

            df = pd.DataFrame([dict(text=txt, label=[lb2id[lb] for lb in lb]) for txt, lb in pairs.items()])
            dset = Dataset.from_pandas(df)
            datasets.set_progress_bar_enabled(False)
            map_args = dict(batched=True, batch_size=64, num_proc=os.cpu_count(), remove_columns=['text'])
            dset = dset.map(tokenize, **map_args)
            datasets.set_progress_bar_enabled(True)
            gen = group_n(range(len(dset)), n=bsz)
            n_ba = math.ceil(n_txt / bsz)

            logger.info(f'Evaluating... ')
            logger_fl.info(f'Evaluating... ')
            it = tqdm(gen, desc=dnm, unit='ba', total=n_ba)
            for i, idxs in enumerate(it):
                inputs = dset[idxs]
                labels = inputs.pop('label')
                inputs = {k: torch.tensor(v, device=device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                preds = torch.argmax(outputs[0], dim=1)
                for i_, (pred, lbs) in enumerate(zip(preds, labels), start=i*bsz):
                    arr_preds[i_] = pred = pred.item()
                    arr_labels[i_] = pred if pred in lbs else lbs[0]
            args = dict(
                zero_division=0, target_names=list(lb2id.keys()), labels=list(range(len(lb2id))), output_dict=True
            )  # disables warning
            df, acc = eval_res2df(arr_labels, arr_preds, report_args=args, pretty=False)
            logger.info(f'{pl.i(dnm)} Classification Accuracy: {pl.i(acc)}')
            logger_fl.info(f'{dnm} Classification Accuracy: {acc}')
            out = os_join(output_path, f'{dnm}.csv')
            df.to_csv(out)
            logger.info(f'{pl.i(dnm)} Eval CSV written to {pl.i(out)}')
            logger_fl.info(f'{dnm} Eval CSV written to {out}')
