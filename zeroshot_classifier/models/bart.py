import os
import pickle
from os.path import join as os_join
from typing import List

import numpy as np
import pandas as pd
import torch.cuda
from sklearn.metrics import classification_report
from transformers import pipeline
from tqdm.auto import tqdm

from zeroshot_classifier.util.load_data import get_datasets

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util


logger = get_logger('BART')


def evaluate(
        model_name: str = 'facebook/bart-large-mnli', domain: str = 'in', dataset_name: str = 'all',
        split_index: int = None, n_splits: int = None
):
    bsz = 32

    all_dset = dataset_name == 'all'
    if not all_dset:
        _dom = sconfig(f'UTCD.datasets.{dataset_name}.domain')
        if domain is not None:
            domain = _dom
        else:
            assert domain == _dom
    if all_dset:
        dataset_names = utcd_util.get_dataset_names(domain)
    else:
        dataset_names = [dataset_name]

    output_dir_nm = f'{now(for_path=True)}_Zeroshot-BART'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))
    os.makedirs(output_path, exist_ok=True)

    is_split = split_index is not None and n_splits is not None
    split_str, split_str_c = None, None
    if is_split:
        assert 0 <= split_index < n_splits
        if dataset_name == 'all':
            raise ValueError(f'Splitting intended for single dataset only')
        split_str = f'{split_index+1}_{n_splits}'
        split_str_c = f'{pl.i(split_index)}/{pl.i(n_splits)}'

    log_fnm = f'{now(for_path=True)}_BART_{domain}_{dataset_name}'
    if is_split:
        log_fnm = f'{log_fnm}_{split_str}'
    log_fnm = f'{log_fnm}_Eval'
    logger_fl = get_logger('BART Eval', kind='file-write', file_path=os_join(output_path, f'{log_fnm}.log'))

    d_log = dict(
        model_name=model_name, domain=domain, dataset_names=dataset_names, batch_size=bsz, output_path=output_path,
        split_index=split_index, n_splits=n_splits
    )
    logger.info(f'Evaluating GPT3 model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating BART model w/ {d_log}... ')

    device = 0 if torch.cuda.is_available() else -1  # See `transformers::pipelines::base`
    model = pipeline('zero-shot-classification', model=model_name, device=device)

    data = get_datasets(domain=domain, dataset_names=dataset_names)

    split = 'test'
    for dnm in dataset_names:  # loop through all datasets
        dset = data[dnm]
        pairs = dset[split]
        d_info = sconfig(f'UTCD.datasets.{dnm}.splits.{split}')
        n_txt, label_options = d_info['n_text'], d_info['labels']
        d_log = {'#text': n_txt, '#label': len(label_options), 'labels': label_options}
        logger.info(f'Evaluating {pl.i(dnm)} w/ {pl.i(d_log)}...')

        lb2id = {lb: idx for idx, lb in enumerate(label_options)}

        if is_split:
            it_txts = iter(split_n(pairs.keys(), n=n_splits))
            txts = None
            for i in range(n_splits):
                txts = next(it_txts)
                if i == split_index:
                    break
            n_txt = len(txts)
            logger.info(f'Loaded split {split_str_c} w/ {pl.i(n_txt)} texts...')
            logger_fl.info(f'Loaded split {split_str} w/ {n_txt} texts...')
        else:
            txts = pairs.keys()
        txts = (txt for txt in txts)
        trues, preds = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
        it = tqdm(model(txts, label_options, batch_size=bsz), desc=f'Evaluating {pl.i(dnm)}', total=n_txt)
        for i, out in enumerate(it):
            txt, labels, scores = out['sequence'], out['labels'], out['scores']
            idx_pred = max(enumerate(scores), key=lambda x: x[1])[0]  # Index of the highest score
            pred = lb2id[labels[idx_pred]]
            lbs_true = [lb2id[lb] for lb in pairs[txt]]
            if pred in lbs_true:
                preds[i] = trues[i] = pred
            else:
                preds[i], trues[i] = -1, lbs_true[0]
        if is_split:
            fnm = f'{now(for_path=True)}_{dnm}_split_{split_str} predictions.pkl'
            path = os_join(output_path, fnm)
            with open(path, 'wb') as f:
                pickle.dump(dict(trues=trues, preds=preds), f)
            logger.info(f'Partial predictions saved to {pl.i(path)}')
            logger_fl.info(f'Partial predictions saved to {path}')
        else:
            args = dict(
                labels=[-1, *range(len(label_options))], target_names=['Label not in dataset', *label_options],
                zero_division=0, output_dict=True
            )
            report = classification_report(trues, preds, **args)
            acc = f'{report["accuracy"]:.3f}'
            logger.info(f'{pl.i(dnm)} accuracy: {pl.i(acc)}')
            logger_fl.info(f'{dnm} accuracy: {acc}')

            path = os_join(output_path, f'{dnm}.csv')
            pd.DataFrame(report).transpose().to_csv(path)


def merge_splits_and_evaluate(domain: str = 'in', dataset_name: str = None, paths: List[str] = None):
    trues, preds = [], []
    for p in paths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        trues.append(d['trues'])
        preds.append(d['preds'])
    trues, preds = np.concatenate(trues), np.concatenate(preds)

    d_info = sconfig(f'UTCD.datasets.{dataset_name}.splits.test')
    n_txt, label_options = d_info['n_text'], d_info['labels']
    assert trues.size == preds.size == n_txt

    args = dict(
        labels=[-1, *range(len(label_options))], target_names=['Label not in dataset', *label_options],
        zero_division=0, output_dict=True
    )
    report = classification_report(trues, preds, **args)
    acc = f'{report["accuracy"]:.3f}'
    logger.info(f'{pl.i(dataset_name)} accuracy: {pl.i(acc)}')

    output_dir_nm = f'{now(for_path=True)}_Zeroshot-BART'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))
    os.makedirs(output_path, exist_ok=True)
    path = os_join(output_path, f'{dataset_name}.csv')
    df = pd.DataFrame(report).transpose()
    df.to_csv(path)
    return df


if __name__ == '__main__':
    # evaluate(domain='in')

    # in the order of #text
    # evaluate(domain='out', dataset_name='finance_sentiment')
    # evaluate(domain='out', dataset_name='snips')
    # evaluate(domain='out', dataset_name='banking77')
    # evaluate(domain='out', dataset_name='patent')
    # evaluate(domain='out', dataset_name='multi_eurlex')
    # evaluate(domain='out', dataset_name='nlu_evaluation')
    # evaluate(domain='out', dataset_name='yelp')
    # evaluate(domain='out', dataset_name='amazon_polarity')

    def chore_merge_splits():
        dir_nms = [
            '2022-10-19_04-07-56_Zeroshot-BART',
            '2022-10-19_04-10-14_Zeroshot-BART',
            '2022-10-19_04-12-11_Zeroshot-BART',
            '2022-10-19_04-14-26_Zeroshot-BART'
        ]
        fnms = [
            '2022-10-19_16-09-25_consumer_finance_split_1_4 predictions',
            '2022-10-19_17-13-13_consumer_finance_split_2_4 predictions',
            '2022-10-19_19-12-10_consumer_finance_split_3_4 predictions',
            '2022-10-19_20-55-40_consumer_finance_split_4_4 predictions'
        ]
        paths = [
            os_join(u.eval_path, dir_nm, '22-10-19_out-of-domain', f'{fnm}.pkl') for dir_nm, fnm in zip(dir_nms, fnms)
        ]
        merge_splits_and_evaluate(domain='out', dataset_name='consumer_finance', paths=paths)
    chore_merge_splits()

    # evaluate(domain='out', dataset_name='consumer_finance', split_index=3, n_splits=4)
