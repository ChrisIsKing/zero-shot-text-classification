import os
import math
import random
from os.path import join as os_join
from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, models, losses, util as sbert_util
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util.load_data import (
    get_datasets, binary_cls_format, in_domain_data_path, out_of_domain_data_path
)
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.models.architecture import BiEncoder


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
    parser_train.add_argument('--model_init', type=str, default='bert-base-uncased')
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--learning_rate', type=float, default=2e-5)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    # set test arguments
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')

    return parser.parse_args()


if __name__ == "__main__":
    seed = sconfig('random-seed')

    NORMALIZE_ASPECT = True

    args = parse_args()
    cmd = args.command
    logger = get_logger(f'{MODEL_NAME} {args.command.capitalize()}')

    if cmd == 'train':
        output_path, sampling, mode = args.output, args.sampling, args.mode
        lr, bsz, n_ep = args.learning_rate, args.batch_size, args.epochs
        model_init = args.model_init

        n = None
        # n = 64

        dset_args = dict(normalize_aspect=seed) if NORMALIZE_ASPECT else dict()
        data = get_datasets(domain='in', n_sample=n, **dset_args)
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == 'in']
        logger.info(f'Loading datasets {pl.i(dataset_names)} for training... ')
        train = []
        val = []
        test = []
        it = tqdm(dataset_names, desc='Formatting into Binary CLS')
        for dataset_name in it:
            dset = data[dataset_name]
            args = dict(dataset_name=dataset_name, sampling=sampling, mode=mode)
            for split, ds in zip(['train', 'val', 'test'], [train, val, test]):
                it.set_postfix(dnm=f'{pl.i(dataset_name)}-{pl.i(split)}')
                ds.extend(binary_cls_format(dset, **args, split=split))

        # seq length for consistency w/ `binary_bert` & `sgd`
        word_embedding_model = models.Transformer(model_init, max_seq_length=512)
        add_tok_arg = utcd_util.get_add_special_tokens_args(word_embedding_model.tokenizer, train_strategy=mode)
        if add_tok_arg:
            logger.info(f'Adding special tokens {pl.i(add_tok_arg)} to tokenizer... ')
            word_embedding_model.tokenizer.add_special_tokens(special_tokens_dict=add_tok_arg)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        model = BiEncoder(modules=[word_embedding_model, pooling_model])

        random.seed(seed)
        random.shuffle(train)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=bsz)
        val_dataloader = DataLoader(val, shuffle=False, batch_size=bsz)
        train_loss = losses.CosineSimilarityLoss(model)

        warmup_steps = math.ceil(len(train_dataloader) * n_ep * 0.1)  # 10% of train data for warm-up
        d_log = {
            '#data': len(train), 'learning_rate': lr, 'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps
        }
        logger.info(f'Launched training with {pl.i(d_log)}... ')

        output_path = map_model_output_path(
            model_name=MODEL_NAME, output_path=output_path,
            mode=mode, sampling=sampling, normalize_aspect=NORMALIZE_ASPECT
        )
        logger.info(f'Model will be saved to {pl.i(output_path)}')

        transformers.set_seed(seed)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            val_dataloader=val_dataloader,
            epochs=n_ep,
            optimizer_params=dict(lr=lr),
            warmup_steps=warmup_steps,
            output_path=output_path
        )
    elif cmd == 'test':
        split = 'test'
        mode, domain, model_path = args.mode, args.domain, args.model_path
        out_path = os_join(model_path, 'eval', domain2eval_dir_nm(domain))
        os.makedirs(out_path, exist_ok=True)

        dataset_names = utcd_util.get_dataset_names(domain)
        data = get_datasets(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        model = SentenceTransformer(args.model_path)
        md_nm = model.__class__.__qualname__

        bsz = 32
        d_log = dict(model=md_nm, mode=mode, domain=domain, datasets=dataset_names, path=model_path, batch_size=bsz)
        logger = get_logger(f'{MODEL_NAME} Eval')
        logger.info(f'Evaluating {MODEL_NAME} with {pl.i(d_log)} and saving to {pl.i(out_path)}... ')

        for dnm in dataset_names:
            pairs: Dict[str, List[str]] = data[dnm][split]
            label_options = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.labels')
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            mode2map = TrainStrategy2PairMap(train_strategy=mode)

            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            d_log = {'#text': n_txt, '#label': len(label_options), 'labels': label_options}
            logger.info(f'Evaluating {pl.i(dnm)} with {pl.i(d_log)}...')

            txts = [mode2map.map_text(t) for t in pairs.keys()]
            label_options = [mode2map.map_label(lb) for lb in label_options]
            logger.info('Encoding texts...')
            txt_embeds = model.encode(txts, batch_size=bsz, show_progress_bar=True)
            logger.info('Encoding labels...')
            lb_opn_embeds = model.encode(label_options, batch_size=bsz, show_progress_bar=True)

            for i, (_, labels) in enumerate(tqdm(pairs.items(), desc=dnm)):
                scores = [sbert_util.cos_sim(txt_embeds[i], v).item() for v in lb_opn_embeds]
                pred = np.argmax(scores)
                label_ids = [label2id[lb] for lb in labels]
                true = pred if pred in label_ids else label_ids[0]
                arr_preds[i], arr_labels[i] = pred, true
            args = dict(zero_division=0, target_names=label_options, output_dict=True)  # disables warning
            df, acc = eval_res2df(arr_labels, arr_preds, **args)
            logger.info(f'{pl.i(dnm)} Classification Accuracy: {pl.i(acc)}')
            df.to_csv(os_join(out_path, f'{dnm}.csv'))
