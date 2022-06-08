import os
import configparser
from os.path import join as os_join
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from zeroshot_classifier.util.data_path import BASE_PATH, PROJ_DIR, DSET_DIR, PKG_NM, MODEL_DIR


from stefutil import ca, StefConfig, StefUtil


__all__ = ['sconfig', 'u', 'save_fig', 'plot_points', 'map_model_output_path']


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
        model_name: str = None, name: str = None, mode: str = 'vanilla', sampling: str = 'rand',
        normalize_aspect: bool = False
) -> str:
    out = f'{now(for_path=True)}_{model_name}'
    if name:
        out = f'{out}-{name}'
    out = f'{out}-{mode}-{sampling}'
    if normalize_aspect:
        out = f'{out}-aspect-norm'
    return out


def map_model_output_path(
        model_name: str = None, output_path: str = None, mode: str = 'vanilla', sampling: str = 'rand',
        normalize_aspect: bool = False
) -> str:
    def _map(dir_nm):
        return map_model_dir_nm(model_name, dir_nm, mode, sampling, normalize_aspect)
    if output_path:
        paths = output_path.split(os.sep)
        output_dir = _map(paths[-1])
        return os_join(*paths[:-1], output_dir)
    else:
        return os_join(u.model_path, _map(None))


if __name__ == '__main__':
    from icecream import ic

    from stefutil import *

    # ic(sconfig('fine-tune'))

    ic(fmt_num(124439808))

    # process_utcd_dataset()

    # map_ag_news()

    # lg = get_logger('test-lang')
    # ic(lg, type(lg))
