import math
import pickle
import random
from typing import List, Dict
from os.path import join as os_join

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util.load_data import get_datasets, binary_cls_format
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.models.architecture import BinaryBertCrossEncoder
from zeroshot_classifier.models._bert_based_models import HF_MODEL_NAME, parse_args


MODEL_NAME = 'Binary BERT'


if __name__ == '__main__':
    import os

    import numpy as np
    import transformers

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
        model = BinaryBertCrossEncoder(md_nm, num_labels=2, automodel_args=dict(ignore_mismatched_sizes=True))

        add_tok_arg = utcd_util.get_add_special_tokens_args(model.tokenizer, train_strategy=mode)
        if add_tok_arg:
            logger.info(f'Adding special tokens {pl.i(add_tok_arg)} to tokenizer... ')
            logger_fl.info(f'Adding special tokens {pl.nc(add_tok_arg)} to tokenizer... ')
            model.tokenizer.add_special_tokens(special_tokens_dict=add_tok_arg)
            model.model.resize_token_embeddings(len(model.tokenizer))

        transformers.logging.set_verbosity_error()  # disables `longest_first` warning
        random.seed(seed)
        random.shuffle(train)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=bsz)
        val_dataloader = DataLoader(val, shuffle=False, batch_size=bsz)
        warmup_steps = math.ceil(len(train_dataloader) * n_ep * 0.1)  # 10% of train data for warm-up

        d_log = {
            '#data': len(train), 'learning_rate': lr, 'batch size': bsz, 'epochs': n_ep, 'warmup steps': warmup_steps,
            'best_model_metric': best_metric, 'output path': output_path
        }
        logger.info(f'Training w/ {pl.i(d_log)}... ')
        logger_fl.info(f'Training w/ {pl.nc(d_log)}... ')

        transformers.set_seed(seed)
        model.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=n_ep,
            optimizer_params=dict(lr=lr),
            warmup_steps=warmup_steps,
            output_path=output_path,
            logger_fl=logger_fl,
            best_model_metric=best_metric
        )
    elif cmd == 'test':
        WITH_EVAL_LOSS = False
        mode, domain, model_name_or_path, bsz = args.mode, args.domain, args.model_name_or_path, args.batch_size
        split = 'test'

        out_path = os_join(u.eval_path, model_name_or_path, domain2eval_dir_nm(domain))
        os.makedirs(out_path, exist_ok=True)

        data = get_datasets(domain=domain)

        model_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_name_or_path)
        if not os.path.exists(model_path):
            model_path = model_name_or_path  # A huggingface model
        logger.info(f'Loading model from path {pl.i(model_path)}... ')
        model = BinaryBertCrossEncoder(model_path)  # load model

        logger = get_logger(f'{MODEL_NAME} Eval')
        d_log = dict(mode=mode, domain=domain, batch_size=bsz, model_name_or_path=model_name_or_path)
        logger.info(f'Evaluating Binary Bert with {pl.i(d_log)} and saving to {pl.i(out_path)}... ')

        eval_loss: Dict[str, np.array] = dict()  # a sense of how badly the model makes the prediction
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == domain]

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
            it = tqdm(gen, desc=F'Evaluating {pl.i(dnm)}', unit='group', total=math.ceil(n_txt/bsz))
            for i_grp, group in enumerate(it):
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
            df.to_csv(os_join(out_path, f'{dnm}.csv'))

        if WITH_EVAL_LOSS:
            with open(os_join(out_path, 'eval_loss.pkl'), 'wb') as f:
                pickle.dump(eval_loss, f)
