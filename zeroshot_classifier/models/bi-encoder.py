import os
import math
import random
from os.path import join
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, models, losses, evaluation, util
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import u, sconfig
from zeroshot_classifier.util.load_data import get_data, binary_cls_format, in_domain_data_path, out_of_domain_data_path
import zeroshot_classifier.util.utcd as utcd_util


MODEL_NAME = 'Bi-Encoder'


def parse_args():
    # see `binary_bert`
    modes = ['vanilla', 'implicit', 'implicit-on-text-encode-aspect', 'implicit-on-text-encode-sep', 'explicit']

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--output', type=str, default=None)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], default='rand')
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    # set test arguments
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = get_logger(f'{MODEL_NAME} {args.command.capitalize()}')

    seed = sconfig('random-seed')

    NORMALIZE_ASPECT = True

    if args.command == 'train':
        output_path, sampling, mode, bsz, n_ep = args.output, args.sampling, args.mode, args.batch_size, args.epochs

        dset_args = dict(normalize_aspect=seed) if NORMALIZE_ASPECT else dict()
        data = get_data(in_domain_data_path, **dset_args)
        # get keys from data dict
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == 'in']
        logger.info(f'Loading datasets {logi(dataset_names)} for training... ')
        train = []
        test = []
        for dataset_name in dataset_names:
            dset = data[dataset_name]
            train += binary_cls_format(dset, name=dataset_name, sampling=sampling, mode=mode)
            test += binary_cls_format(dset, train=False, mode=mode)

        # seq length for consistency w/ `binary_bert` & `sgd`
        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=512)
        word_embedding_model.tokenizer.add_special_tokens(dict(eos_token=utcd_util.EOT_TOKEN))
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        # default to mean-pooling
        pooling_model = models.Pooling(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        random.seed(seed)
        random.shuffle(train)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=bsz)
        train_loss = losses.CosineSimilarityLoss(model)

        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test, name='UTCD-test')

        warmup_steps = math.ceil(len(train_dataloader) * n_ep * 0.1)  # 10% of train data for warm-up
        d_log = {'#data': len(train), 'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps}
        logger.info(f'Launched training with {log_dict(d_log)}... ')

        def dir_nm2dir_nm(nm: str = None):
            out = f'{now(for_path=True)}_{MODEL_NAME}'
            if nm:
                out = f'{out}-{nm}'
            out = f'{out}-{mode}-{sampling}'
            if NORMALIZE_ASPECT:
                out = f'{out}-aspect-norm'
            return out
        if output_path:
            paths = args.output.split(os.sep)
            output_dir = dir_nm2dir_nm(paths[-1])
            output_path = join(*paths[:-1], output_dir)
        else:
            output_path = join(u.model_path, dir_nm2dir_nm())
        logger.info(f'Model will be saved to {logi(output_path)}')
        transformers.set_seed(seed)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=n_ep,
            # internally, passing in an evaluator means after training ends, model not saved...
            evaluator=evaluator,
            warmup_steps=warmup_steps,
            evaluation_steps=100000,
            output_path=output_path
        )
        # hence, make explicit call to save model
        model.save(output_path)
    if args.command == 'test':
        pred_path = join(args.model_path, 'preds/{}/'.format(args.domain))
        result_path = join(args.model_path, 'results/{}/'.format(args.domain))
        Path(pred_path).mkdir(parents=True, exist_ok=True)
        Path(result_path).mkdir(parents=True, exist_ok=True)
        if args.domain == 'in':
            data = get_data(in_domain_data_path)
        elif args.domain == 'out':
            data = get_data(out_of_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())

        # load model
        model = SentenceTransformer(args.model_path)

        # loop through all datasets
        for dataset_name in datasets:
            examples = data[dataset_name]["test"]
            labels = data[dataset_name]['labels'] if args.mode == 'vanilla' else [
                '{} {}'.format(label, data[dataset_name]['aspect']) for label in data[dataset_name]['labels']]
            preds = []
            gold = []
            correct = 0

            example_vectors = model.encode(list(examples.keys()))
            label_vectors = model.encode(labels)

            # loop through each test example
            print("Evaluating dataset: {}".format(dataset_name))
            for index, (text, gold_labels) in enumerate(tqdm(examples.items())):
                if args.mode == 'implicit':
                    gold_labels = [f'{label} {data[dataset_name]["aspect"]}' for label in gold_labels]
                results = [util.cos_sim(example_vectors[index], label_vectors[i]) for i in range(len(labels))]

                # compute which pred is higher
                pred = labels[np.argmax(results)]
                preds.append(pred)

                if pred in gold_labels:
                    correct += 1
                    gold.append(pred)
                else:
                    gold.append(gold_labels[0])

            print('{} Dataset Accuracy = {}'.format(dataset_name, correct / len(examples)))
            report = classification_report(gold, preds, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv('{}/{}.csv'.format(result_path, dataset_name))
