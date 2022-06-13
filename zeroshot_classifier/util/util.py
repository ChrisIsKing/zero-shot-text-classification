import os
import datetime
import configparser
from os.path import join as os_join
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import sklearn
from datasets import load_metric
import matplotlib.pyplot as plt

from stefutil import *
from zeroshot_classifier.util.data_path import BASE_PATH, PROJ_DIR, DSET_DIR, PKG_NM, MODEL_DIR


__all__ = [
    'sconfig', 'u', 'save_fig', 'plot_points',
    'map_model_output_path', 'domain2eval_dir_nm', 'TrainStrategy2PairMap',
    'eval_res2df', 'compute_metrics'
]


sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
u.plot_path = os_join(BASE_PATH, PROJ_DIR, 'plot')
save_fig = u.save_fig

for d in sconfig('check-arg'):
    ca.cache_mismatch(**d)


def plot_points(arr, **kwargs):
    """
    :param arr: Array of 2d points to plot
    :param kwargs: Arguments are forwarded to `matplotlib.axes.Axes.plot`
    """
    arr = np.asarray(arr)
    kwargs_ = dict(marker='.', lw=0.5, ms=1, c='orange')
    kwargs = {**kwargs_, **kwargs}  # python3.6 compatibility
    plt.plot(arr[:, 0], arr[:, 1], **kwargs)


def config_parser2dict(conf: configparser.ConfigParser) -> Dict:
    return {sec: dict(conf[sec]) for sec in conf.sections()}


def map_model_dir_nm(
        model_name: str = None, name: str = None, mode: Optional[str] = 'vanilla',
        sampling: Optional[str] = 'rand', normalize_aspect: bool = False
) -> str:
    out = f'{now(for_path=True)}_{model_name}'
    if name:
        out = f'{out}-{name}'
    if mode:
        out = f'{out}-{mode}'
    if sampling:
        out = f'{out}-{sampling}'
    if normalize_aspect:
        out = f'{out}-aspect-norm'
    return out


def map_model_output_path(
        model_name: str = None, output_path: str = None, mode: Optional[str] = 'vanilla',
        sampling: Optional[str] = 'rand', normalize_aspect: bool = False
) -> str:
    def _map(dir_nm):
        return map_model_dir_nm(model_name, dir_nm, mode, sampling, normalize_aspect)
    if output_path:
        paths = output_path.split(os.sep)
        output_dir = _map(paths[-1])
        return os_join(*paths[:-1], output_dir)
    else:
        return os_join(u.proj_path, u.model_dir, _map(None))


def domain2eval_dir_nm(domain: str = 'in'):
    domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
    date = datetime.datetime.now().strftime('%m.%d.%Y')
    date = date[:-4] + date[-2:]  # 2-digit year
    return f'{domain_str}, {date}'


class TrainStrategy2PairMap:
    sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')
    aspect2aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')

    def __init__(self, train_strategy: str = 'vanilla'):
        self.train_strategy = train_strategy
        ca(training_strategy=train_strategy)

    def __call__(self, aspect: str = None):
        if self.train_strategy in ['vanilla', 'explicit']:
            def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                return [[txt, lb] for lb in lbs]
        elif self.train_strategy == 'implicit':
            def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                return [[txt, f'{lb} {aspect}'] for lb in lbs]
        elif self.train_strategy == 'implicit-on-text-encode-aspect':
            def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                return [[f'{TrainStrategy2PairMap.aspect2aspect_token[aspect]} {txt}', lb] for lb in lbs]
        else:
            assert self.train_strategy == 'implicit-on-text-encode-sep'

            def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                return [[f'{aspect} {TrainStrategy2PairMap.sep_token} {txt}', lb] for lb in lbs]
        return txt_n_lbs2query

    def map_label(self, label: str, aspect: str = None):
        return f'{label} {aspect}' if self.train_strategy == 'implicit' else label

    def map_text(self, text: str, aspect: str = None):
        if self.train_strategy == 'implicit-on-text-encode-aspect':
            return f'{TrainStrategy2PairMap.aspect2aspect_token[aspect]} {text}'
        elif self.train_strategy == 'implicit-on-text-encode-sep':
            return f'{aspect} {TrainStrategy2PairMap.sep_token} {text}'
        else:
            return text


def eval_res2df(labels: Iterable, preds: Iterable, report_args: Dict = None) -> Tuple[pd.DataFrame, float]:
    report = sklearn.metrics.classification_report(labels, preds, **(report_args or dict()))
    if 'accuracy' in report:
        acc = report['accuracy']
    else:
        vals = [v for k, v in report['micro avg'].items() if k != 'support']
        assert all(v == vals[0] for v in vals)
        acc = vals[0]
    return pd.DataFrame(report).transpose(), round(acc, 3)


def compute_metrics(eval_pred):
    if not hasattr(compute_metrics, 'acc'):
        compute_metrics.acc = load_metric('accuracy')
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return dict(acc=compute_metrics.acc.compute(predictions=preds, references=labels)['accuracy'])


if __name__ == '__main__':
    from icecream import ic

    from stefutil import *

    # ic(sconfig('fine-tune'))

    ic(fmt_num(124439808))

    # process_utcd_dataset()

    # map_ag_news()

    # lg = get_logger('test-lang')
    # ic(lg, type(lg))
