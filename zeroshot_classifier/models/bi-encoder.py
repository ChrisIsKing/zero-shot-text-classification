import os
import math
import random
from os.path import join as os_join
from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, models, losses, evaluation, util as sbert_util
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *
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
        for dnm in dataset_names:
            dset = data[dnm]
            train += binary_cls_format(dset, name=dnm, sampling=sampling, mode=mode)
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

        output_path = map_model_output_path(
            model_name=MODEL_NAME, output_path=output_path,
            mode=mode, sampling=sampling, normalize_aspect=NORMALIZE_ASPECT
        )
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
        mic(os.listdir(output_path))
    if args.command == 'test':
        split = 'test'
        mode, domain, model_path = args.mode, args.domain, args.model_path
        out_path = os_join(model_path, 'eval', domain2eval_dir_nm(domain))
        os.makedirs(out_path, exist_ok=True)
        logger = get_logger(f'{MODEL_NAME} Eval')
        d_log = dict(mode=mode, domain=domain, path=model_path)
        logger.info(f'Evaluating Binary Bert with {log_dict(d_log)} and saving to {logi(out_path)}... ')

        data = get_data(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        dataset_names = get_dataset_names(domain)

        model = SentenceTransformer(args.model_path)

        for dnm in dataset_names:
            pairs: Dict[str, List[str]] = data[dnm][split]
            label_options = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.labels')
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            txt_n_lbs2query = TrainStrategy2PairMap(train_strategy=mode)(aspect=data[dnm]['aspect'])

            labels = data[dnm]['labels'] if args.mode == 'vanilla' else [
                '{} {}'.format(label, data[dnm]['aspect']) for label in data[dnm]['labels']]
            preds = []
            gold = []
            correct = 0

            example_vectors = model.encode(list(pairs.keys()))
            label_vectors = model.encode(labels)

            # loop through each test example
            print("Evaluating dataset: {}".format(dnm))
            for index, (text, gold_labels) in enumerate(tqdm(pairs.items())):
                if args.mode == 'implicit':
                    gold_labels = [f'{label} {data[dnm]["aspect"]}' for label in gold_labels]
                results = [sbert_util.cos_sim(example_vectors[index], label_vectors[i]) for i in range(len(labels))]

                # compute which pred is higher
                pred = labels[np.argmax(results)]
                preds.append(pred)

                if pred in gold_labels:
                    correct += 1
                    gold.append(pred)
                else:
                    gold.append(gold_labels[0])

            args = dict(zero_division=0, target_names=label_options, output_dict=True)  # disables warning
            df, acc = eval_res2df(gold, preds, **args)
            logger.info(f'{logi(dnm)} Classification Accuracy: {logi(acc)}')
            df.to_csv(os_join(out_path, f'{dnm}.csv'))
            exit(1)
