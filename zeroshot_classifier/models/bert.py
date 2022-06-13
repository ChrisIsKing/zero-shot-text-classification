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
    parser_test.add_argument('--path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    import os

    import transformers

    args = parse_args()

    seed = sconfig('random-seed')
    NORMALIZE_ASPECT = True

    if args.command == 'train':
        logger = get_logger(f'{MODEL_NAME} Train')
        dataset_name, domain = args.dataset, args.domain
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'

        dset_args = dict(normalize_aspect=seed) if NORMALIZE_ASPECT else dict()
        data = get_data(in_domain_data_path, **dset_args)
        if dataset_name == 'all':
            train_dset, test_dset, labels = seq_cls_format(data, all=True)
        else:
            train_dset, test_dset, labels = seq_cls_format(data[dataset_name])
        d_log = {'#train': len(train_dset), '#test': len(test_dset), 'labels': labels}
        logger.info(f'Loaded {logi(domain_str)} dataset {logi(dataset_name)} with {log_dict(d_log)} ')

        num_labels = len(labels)
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(HF_MODEL_NAME, return_dict=True, num_labels=num_labels)
        tokenizer.add_special_tokens(dict(eos_token=utcd_util.EOT_TOKEN))  # end-of-turn for SGD
        model.resize_token_embeddings(len(tokenizer))

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        train_dset = Dataset.from_pandas(pd.DataFrame(train_dset))
        test_dset = Dataset.from_pandas(pd.DataFrame(test_dset))
        train_dset = train_dset.map(tokenize_function, batched=True)
        test_dset = test_dset.map(tokenize_function, batched=True)

        bsz, n_ep = 16, 3
        warmup_steps = math.ceil(len(train_dset) * n_ep * 0.1)  # 10% of train data for warm-up

        dir_nm = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), output_path=f'{domain}-{dataset_name}', mode=None,
            sampling=None, normalize_aspect=NORMALIZE_ASPECT
        )
        output_path = os_join(utcd_util.get_output_base(), u.proj_dir, u.model_dir, dir_nm)
        proj_output_path = os_join(u.model_path, dir_nm, 'trained')
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
        logger = get_logger(f'{MODEL_NAME} Eval')

        if args.domain == "in":
            data = get_data(in_domain_data_path)
        else:
            data = get_data(out_of_domain_data_path)
        
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(args.path)
        model.to("cuda")

        def tokenize(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        if args.dataset == "all":
            dataset = data
            train, test, labels = seq_cls_format(dataset, all=True)
        
        index = 0

        for name, dataset in data.items():
            dataset_len = len(dataset['test'])

            test_dset = test[index:index + dataset_len]
            index += dataset_len

            test_dset = Dataset.from_pandas(pd.DataFrame(test_dset))
            test_dset = test_dset.map(tokenize, batched=True)

            output_path = './models/{}'.format(args.dataset)

            training_args = TrainingArguments(
                output_dir=output_path,          # output directory
                per_device_train_batch_size=16,  # batch size per device during training
                per_device_eval_batch_size=32,   # batch size for evaluation
                weight_decay=0.01,               # strength of weight decay
                logging_dir='./logs',            # directory for storing logs
                load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
                # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
                logging_steps=100000,               # log & save weights each logging_steps
                save_steps=100000,
                evaluation_strategy="steps",     # evaluate each `logging_steps`
            )

            trainer = Trainer(
                model=model,                         # the instantiated Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                eval_dataset=test_dset,          # evaluation dataset
                compute_metrics=compute_metrics,     # the callback that computes metrics of interest
            )

            print("Evaluation Metrics for {} dataset".format(name))
            print(trainer.evaluate())
