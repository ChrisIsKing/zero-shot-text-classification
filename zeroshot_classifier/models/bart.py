import os
from os.path import join as os_join

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import pipeline
from tqdm.auto import tqdm

from zeroshot_classifier.util.load_data import get_datasets

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util


logger = get_logger('BART')


def evaluate(model_name: str = 'facebook/bart-large-mnli', domain: str = 'in'):
    bsz = 32

    output_dir_nm = f'{now(for_path=True)}_Zeroshot-BART'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))
    os.makedirs(output_path, exist_ok=True)

    log_fnm = f'{now(for_path=True)}_BART_{domain}_Eval'
    logger_fl = get_logger('BART Eval', typ='file-write', file_path=os_join(output_path, f'{log_fnm}.log'))

    d_log = dict(model_name=model_name, domain=domain, batch_size=bsz, output_path=output_path)
    logger.info(f'Evaluating GPT3 model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating BART model w/ {d_log}... ')

    model = pipeline('zero-shot-classification', model=model_name)

    data = get_datasets(domain=domain)
    dataset_names = utcd_util.get_dataset_names(domain)

    split = 'test'
    for dnm in dataset_names:  # loop through all datasets
        if dnm != 'emotion':  # TODO: debugging
            continue
        dset = data[dnm]
        pairs = dset[split]
        d_info = sconfig(f'UTCD.datasets.{dnm}.splits.{split}')
        n_txt, label_options = d_info['n_text'], d_info['labels']
        d_log = {'#text': n_txt, '#label': len(label_options), 'labels': label_options}
        logger.info(f'Evaluating {pl.i(dnm)} w/ {pl.i(d_log)}...')

        lb2id = {lb: idx for idx, lb in enumerate(label_options)}
        trues, preds = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
        txts = (txt for txt in pairs.keys())
        it = tqdm(model(txts, label_options, batch_size=bsz), desc=f'Evaluating {pl.i(dnm)}', total=n_txt)
        for i, out in enumerate(it):
            txt, labels, scores = out['sequence'], out['labels'], out['scores']
            # mic(txt, labels, scores)

            idx_pred = max(enumerate(scores), key=lambda x: x[1])[0]  # Index of highest score
            pred = lb2id[labels[idx_pred]]
            # pred = max(zip(labels, scores), key=lambda x: x[1])[0]
            lbs_true = [lb2id[lb] for lb in pairs[txt]]
            # mic(pred, trues)
            if pred in label_options:
                preds[i] = trues[i] = pred
            else:
                preds[i], trues[i] = -1, lbs_true[0]
            # exit(1)
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
        exit(1)


if __name__ == '__main__':
    evaluate(domain='in')
