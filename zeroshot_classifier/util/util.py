import os
import math
import json
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
    'on_great_lakes', 'get_base_path',
    'map_model_dir_nm', 'map_model_output_path', 'domain2eval_dir_nm', 'TrainStrategy2PairMap',
    'eval_res2df', 'compute_metrics'
]


logger = get_logger('Util')


config_path = os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')
if not os.path.exists(config_path):
    from zeroshot_classifier.util.config import config_dict
    logger.info(f'Writing config file at {pl.i(config_path)}')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

sconfig = StefConfig(config_file=config_path).__call__
u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
u.plot_path = os_join(BASE_PATH, PROJ_DIR, 'plot')
save_fig = u.save_fig

for _d in sconfig('check-arg'):
    ca.cache_mismatch(**_d)


def plot_points(arr, **kwargs):
    """
    :param arr: Array of 2d points to plot
    :param kwargs: Arguments are forwarded to `matplotlib.axes.Axes.plot`
    """
    arr = np.asarray(arr)
    kwargs_ = dict(marker='.', lw=0.5, ms=1, c='orange')
    kwargs = {**kwargs_, **kwargs}  # python3.6 compatibility
    plt.plot(arr[:, 0], arr[:, 1], **kwargs)


def on_great_lakes():
    return 'arc-ts' in get_hostname()


def get_base_path():
    # For remote machines, save heavy-duty data somewhere else to save `/home` disk space
    hnm = get_hostname()
    if 'clarity' in hnm:  # Clarity lab
        return '/data'
    elif on_great_lakes():  # Great Lakes; `profmars0` picked arbitrarily among [`profmars0`, `profmars1`]
        # Per https://arc.umich.edu/greatlakes/user-guide/
        return os_join('/scratch', 'profmars_root', 'profmars0', 'stefanhg')
    else:
        return BASE_PATH


def config_parser2dict(conf: configparser.ConfigParser) -> Dict:
    return {sec: dict(conf[sec]) for sec in conf.sections()}


def map_model_dir_nm(
        model_name: str = None, name: str = None, mode: Optional[str] = 'vanilla',
        sampling: Optional[str] = 'rand', normalize_aspect: bool = False
) -> str:
    out = f'{now(for_path=True)}_{model_name}'
    if name:
        out = f'{out}_{name}'
    d = dict()
    if mode:  # see config::training.strategies
        nms = mode.split('-')
        if len(nms) == 1:
            d['md'] = mode[:3]
        else:
            nf, nl = nms[0], nms[-1]
            d['md'] = f'{nf[:3]}-{nl[:3]}'
    if sampling:
        d['sp'] = sampling[0]
    if normalize_aspect:
        d['na'] = 'T'
    if d:
        out = f'{out}_{pl.pa(d)}'
    return out


def map_model_output_path(
        model_name: str = None, output_path: str = None, output_dir: str = None, mode: Optional[str] = 'vanilla',
        sampling: Optional[str] = 'rand', normalize_aspect: bool = False
) -> str:
    def _map(dir_nm_):
        return map_model_dir_nm(model_name, dir_nm_, mode, sampling, normalize_aspect)

    assert (output_path or output_dir) and not (output_path and output_dir)  # sanity check mutually exclusive
    if output_path:
        paths = output_path.split(os.sep)
        output_dir = _map(paths[-1])
        return os_join(*paths[:-1], output_dir)
    else:
        dir_nm = _map(None)
        if output_dir:
            dir_nm = f'{dir_nm}_{output_dir}'
        return os_join(get_base_path(), u.proj_dir, u.model_dir, dir_nm)


def domain2eval_dir_nm(domain: str = 'in'):
    domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
    date = now(fmt='short-date')
    return f'{date}_{domain_str}'


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
        if self.train_strategy == 'implicit':
            assert aspect is not None
            return f'{label} {aspect}'
        else:
            return label

    def map_text(self, text: str, aspect: str = None):
        if self.train_strategy in ['implicit-on-text-encode-aspect', 'implicit-on-text-encode-sep']:
            assert aspect is not None
            if self.train_strategy == 'implicit-on-text-encode-aspect':
                return f'{TrainStrategy2PairMap.aspect2aspect_token[aspect]} {text}'
            else:
                return f'{aspect} {TrainStrategy2PairMap.sep_token} {text}'
        else:
            return text


def eval_res2df(labels: Iterable, preds: Iterable, report_args: Dict = None, pretty: bool = True) -> Tuple[pd.DataFrame, float]:
    report = sklearn.metrics.classification_report(labels, preds, **(report_args or dict()))
    if 'accuracy' in report:
        acc = report['accuracy']
    else:
        vals = [v for k, v in report['micro avg'].items() if k != 'support']
        assert all(math.isclose(v, vals[0], abs_tol=1e-8) for v in vals)
        acc = vals[0]
    return pd.DataFrame(report).transpose(), round(acc, 3) if pretty else acc


def compute_metrics(eval_pred):
    if not hasattr(compute_metrics, 'acc'):
        compute_metrics.acc = load_metric('accuracy')
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return dict(acc=compute_metrics.acc.compute(predictions=preds, references=labels)['accuracy'])


if __name__ == '__main__':
    from stefutil import *

    def check_gl():
        mic(on_great_lakes())
        mic(get_base_path())
    check_gl()
