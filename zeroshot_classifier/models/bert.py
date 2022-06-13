import math
from os.path import join as os_join
from argparse import ArgumentParser

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.util.load_data import get_data, seq_cls_format, in_domain_data_path, out_of_domain_data_path


MODEL_NAME = 'BERT Seq CLS'
HF_MODEL_NAME = 'bert-base-uncased'


def parse_args():
    parser = ArgumentParser()

    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    parser_train.add_argument('--dataset', type=str, default='all')
    parser_train.add_argument('--domain', type=str, choices=['in', 'out'], required=True)

    parser_test.add_argument('--dataset', type=str, default='all')
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--model_path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    import os

    import transformers
    import datasets

    args = parse_args()

    seed = sconfig('random-seed')
    NORMALIZE_ASPECT = True

    if args.command == 'train':
        logger = get_logger(f'{MODEL_NAME} Train')
        dataset_name, domain = args.dataset, args.domain
        ca(dataset_domain=domain)
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'

        dset_args = dict(domain=domain)
        if NORMALIZE_ASPECT:
            dset_args['normalize_aspect'] = seed
        data = get_data(in_domain_data_path if domain == 'in' else out_of_domain_data_path, **dset_args)
        if dataset_name == 'all':
            train_dset, test_dset, labels = seq_cls_format(data, all=True)
        else:
            train_dset, test_dset, labels = seq_cls_format(data[dataset_name])
        d_log = {'#train': len(train_dset), '#test': len(test_dset), 'labels': list(labels.keys())}
        logger.info(f'Loaded {logi(domain_str)} datasets {logi(dataset_name)} with {log_dict(d_log)} ')

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

        bsz, n_ep = 16, 3
        warmup_steps = math.ceil(len(train_dset) * n_ep * 0.1)  # 10% of train data for warm-up

        dir_nm = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), output_path=f'{domain}-{dataset_name}', mode=None,
            sampling=None, normalize_aspect=NORMALIZE_ASPECT
        )
        output_path = os_join(utcd_util.get_output_base(), u.proj_dir, u.model_dir, dir_nm)
        proj_output_path = os_join(u.base_path, u.proj_dir, u.model_path, dir_nm, 'trained')
        mic(dir_nm, proj_output_path)
        d_log = {'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps, 'save path': output_path}
        logger.info(f'Launched training with {log_dict(d_log)}... ')

        training_args = TrainingArguments(  # TODO: learning rate
            output_dir=output_path,
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
        mic(trainer.evaluate())
        trainer.save_model(proj_output_path)
        tokenizer.save_pretrained(proj_output_path)
        os.listdir(proj_output_path)

    if args.command == 'test':
        dataset_name, domain, model_path = args.dataset, args.domain, args.model_path
        bsz = 32
        split = 'test'
        assert dataset_name == 'all'
        dataset_names = utcd_util.get_dataset_names(domain)
        output_path = os_join(model_path, 'eval')
        lg_nm = f'{MODEL_NAME} Eval'
        logger = get_logger(lg_nm)
        lg_fl = os_join(output_path, f'{now(for_path=True)}_{lg_nm}, dom={domain}.log')
        logger_fl = get_logger(lg_nm, typ='file-write', file_path=lg_fl)
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        logger.info(f'Evaluating {logi(domain_str)} datasets {log_list(dataset_names)} on model {logi(model_path)}... ')
        logger_fl.info(f'Evaluating {domain_str} datasets {dataset_names} on model {model_path}... ')

        data = get_data(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        _, dset_formatted, labels = seq_cls_format(data, all=True)  # Need to pass in all data for `label_map`

        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        i_dset_strt = 0
        for dnm, dset in data.items():  # ordering matters since selecting formatted samples
        # for dnm in dataset_names:
            dset = dset['test']
            asp = sconfig(f'UTCD.datasets.{dnm}.aspect')
            logger.info(f'Evaluating {logi(asp)} dataset {logi(dnm)}... ')
            logger_fl.info(f'Evaluating {asp} dataset {dnm}... ')

            # dset_ = seq_cls_format(data[dnm])[1]  # test split
            n = len(dset)
            mic(n, sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text'))
            assert n == sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')  # sanity check
            dset_ = dset_formatted[i_dset_strt:i_dset_strt+n]
            i_dset_strt += n

            logger.info(f'Loading {logi(n)} samples... ')
            logger_fl.info(f'Loading {n} samples... ')
            dset_ = Dataset.from_pandas(pd.DataFrame(dset_))
            datasets.set_progress_bar_enabled(False)
            map_args = dict(batched=True, batch_size=64, num_proc=os.cpu_count(), remove_columns=['label_name', 'text'])
            dset_ = dset_.map(tokenize, **map_args)
            datasets.set_progress_bar_enabled(True)
            # TODO: fix multi-label

            logger.info(f'Evaluating... ')
            logger_fl.info(f'Evaluating... ')
            training_args = TrainingArguments(  # Take trainer for eval
                output_dir=output_path,  # Expect nothing to be written to
                per_device_eval_batch_size=bsz
            )
            d_eval = Trainer(
                model=model, args=training_args, eval_dataset=dset_, compute_metrics=compute_metrics
            ).evaluate()
            acc = d_eval['eval_acc']
            logger.info(f'{logi(dnm)} Classification Accuracy: {logi(round(acc, 3))}')
            logger_fl.info(f'{dnm} Classification Accuracy: {acc}')
        mic(i_dset_strt, len(dset_formatted))
        assert i_dset_strt == len(dset_formatted)  # sanity check
