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

        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        md_nm = model.__class__.__qualname__
        logger.info(f'Evaluating {logi(domain_str)} on {logi(md_nm)} and datasets {logi(dataset_names)}... ')

        for dnm in dataset_names:
            pairs: Dict[str, List[str]] = data[dnm][split]
            label_options = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.labels')
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            mode2map = TrainStrategy2PairMap(train_strategy=mode)
            txt_n_lbs2query = mode2map(aspect=data[dnm]['aspect'])

            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            d_log = {'#text': n_txt, '#label': len(label_options), 'labels': label_options}
            logger.info(f'Evaluating {logi(dnm)} with {log_dict(d_log)}...')

            txts = [mode2map.map_text(t) for t in pairs.keys()]
            label_options = [mode2map.map_label(lb) for lb in label_options]
            logger.info('Encoding texts...')
            txt_embeds = model.encode(txts, show_progress_bar=True)
            logger.info('Encoding labels...')
            lb_opn_embeds = model.encode(label_options, show_progress_bar=True)

            for i, (_, labels) in enumerate(tqdm(pairs.items(), desc=dnm)):
                scores = [sbert_util.cos_sim(txt_embeds[i], v).item() for v in lb_opn_embeds]
                pred = np.argmax(scores)
                label_ids = [label2id[lb] for lb in labels]
                true = pred if pred in label_ids else label_ids[0]
                arr_preds[i], arr_labels[i] = pred, true
            args = dict(zero_division=0, target_names=label_options, output_dict=True)  # disables warning
            df, acc = eval_res2df(arr_labels, arr_preds, **args)
            logger.info(f'{logi(dnm)} Classification Accuracy: {logi(acc)}')
            df.to_csv(os_join(out_path, f'{dnm}.csv'))
