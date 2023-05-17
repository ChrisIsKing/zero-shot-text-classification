import os
import math
import random
from os.path import join as os_join
from typing import List, Dict

import numpy as np
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, models, losses, util as sbert_util
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util.load_data import get_datasets, binary_cls_format
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.models.architecture import BiEncoder
from zeroshot_classifier.models._bert_based_models import HF_MODEL_NAME, parse_args


MODEL_NAME = 'Bi-Encoder'


if __name__ == "__main__":
    seed = sconfig('random-seed')

    args = parse_args()
    cmd = args.command
    log_nm = f'{MODEL_NAME} {args.command.capitalize()}'
    logger = get_logger(log_nm)

    if cmd == 'train':
        output_path, output_dir, sampling, mode = args.output, args.output_dir, args.sampling, args.mode
        normalize_aspect = args.normalize_aspect
        lr, bsz, n_ep = args.learning_rate, args.batch_size, args.epochs
        init_model_name_or_path = args.init_model_name_or_path

        # best_metric = 'accuracy'
        best_metric = 'loss'

        output_path = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), output_path=output_path, output_dir=output_dir,
            mode=mode, sampling=sampling, normalize_aspect=normalize_aspect
        )
        logger_fl = get_logger(log_nm, kind='file-write', file_path=os_join(output_path, 'training.log'))

        dset_args = dict(normalize_aspect=seed) if normalize_aspect else dict()
        data = get_datasets(domain='in', **dset_args)
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == 'in']
        logger.info(f'Processing datasets {pl.i(dataset_names)} for training... ')
        logger_fl.info(f'Processing datasets {pl.nc(dataset_names)} for training... ')

        train, val, test = [], [], []
        it = tqdm(dataset_names, desc=f'Formatting into Binary CLS w/ {pl.i(dict(sampling=sampling, mode=mode))}')
        for dataset_name in it:
            dset = data[dataset_name]
            args = dict(sampling=sampling, mode=mode)
            for split, ds in zip(['train', 'val', 'test'], [train, val, test]):
                it.set_postfix(dnm=f'{pl.i(dataset_name)}-{pl.i(split)}')
                ds.extend(binary_cls_format(dset, **args, split=split))

        d_log = dict(init_model_name_or_path=init_model_name_or_path)
        md_nm = init_model_name_or_path
        if mode == 'explicit':
            assert init_model_name_or_path != HF_MODEL_NAME  # sanity check
        if init_model_name_or_path != HF_MODEL_NAME:
            # loading from explicit pre-training local weights,
            # the classification head would be ignored for classifying 3 classes
            path = os_join(get_base_path(), u.proj_dir, u.model_dir, init_model_name_or_path)
            if os.path.exists(path):
                md_nm = path
                d_log['files'] = os.listdir(path)
        logger.info(f'Loading model with {pl.i(d_log)}...')
        logger_fl.info(f'Loading model with {pl.nc(d_log)}...')
        word_embedding_model = models.Transformer(
            init_model_name_or_path, max_seq_length=512, tokenizer_args=dict(use_fast=False)
        )
        add_tok_arg = utcd_util.get_add_special_tokens_args(word_embedding_model.tokenizer, train_strategy=mode)
        if add_tok_arg:
            logger.info(f'Adding special tokens {pl.i(add_tok_arg)} to tokenizer... ')
            logger_fl.info(f'Adding special tokens {pl.nc(add_tok_arg)} to tokenizer... ')
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
            '#data': len(train), 'learning_rate': lr, 'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps,
            'best_model_metric': best_metric, 'output path': output_path
        }
        logger.info(f'Training w/ {pl.i(d_log)}... ')
        logger_fl.info(f'Training w/ {pl.nc(d_log)}... ')

        transformers.set_seed(seed)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            val_dataloader=val_dataloader,
            epochs=n_ep,
            optimizer_params=dict(lr=lr),
            warmup_steps=warmup_steps,
            output_path=output_path,
            logger_fl=logger_fl,
            best_model_metric=best_metric
        )
    elif cmd == 'test':
        split = 'test'
        mode, domain, model_name_or_path, bsz = args.mode, args.domain, args.model_name_or_path, args.batch_size
        out_path = os_join(u.eval_path, model_name_or_path, domain2eval_dir_nm(domain))
        os.makedirs(out_path, exist_ok=True)

        dataset_names = utcd_util.get_dataset_names(domain)
        data = get_datasets(domain=domain)

        model_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_name_or_path)
        if not os.path.exists(model_path):
            model_path = model_name_or_path  # A huggingface model
        logger.info(f'Loading model from path {pl.i(model_path)}... ')
        model = SentenceTransformer(model_path)
        md_nm = model.__class__.__qualname__

        d_log = dict(
            model=md_nm, mode=mode, domain=domain, datasets=dataset_names, model_name_or_path=model_name_or_path,
            batch_size=bsz
        )
        logger = get_logger(f'{MODEL_NAME} Eval')
        logger.info(f'Evaluating {MODEL_NAME} with {pl.i(d_log)} and saving to {pl.i(out_path)}... ')

        for dnm in dataset_names:
            dset = data[dnm]
            pairs: Dict[str, List[str]] = dset[split]
            aspect = dset['aspect']
            label_options = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.labels')
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            mode2map = TrainStrategy2PairMap(train_strategy=mode)
            txts = [mode2map.map_text(t, aspect=aspect) for t in pairs.keys()]
            label_options = [mode2map.map_label(lb, aspect=aspect) for lb in label_options]

            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            d_log = {'#text': n_txt, '#label': len(label_options), 'labels': label_options}
            logger.info(f'Evaluating {pl.i(dnm)} with {pl.i(d_log)}...')

            logger.info('Encoding texts...')
            txt_embeds = model.encode(txts, batch_size=bsz, show_progress_bar=True)
            logger.info('Encoding labels...')
            lb_opn_embeds = model.encode(label_options, batch_size=bsz, show_progress_bar=True)

            for i, (_, labels) in enumerate(tqdm(pairs.items(), desc=f'Evaluating {pl.i(dnm)}')):
                scores = [sbert_util.cos_sim(txt_embeds[i], v).item() for v in lb_opn_embeds]
                pred = np.argmax(scores)
                label_ids = [label2id[lb] for lb in labels]
                true = pred if pred in label_ids else label_ids[0]
                arr_preds[i], arr_labels[i] = pred, true
            args = dict(zero_division=0, target_names=label_options, output_dict=True)  # disables warning
            df, acc = eval_res2df(arr_labels, arr_preds, report_args=args)
            logger.info(f'{pl.i(dnm)} Classification Accuracy: {pl.i(acc)}')
            df.to_csv(os_join(out_path, f'{dnm}.csv'))
