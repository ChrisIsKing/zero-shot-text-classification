"""
Get the text samples that models perform the worst on, and look for insights

See `zeroshot_encoder.baseline.binary_bert` in test mode
"""

import json
from os.path import join as os_join
from typing import List, Tuple, Dict

import numpy as np

from stefutil import *
from zeroshot_encoder.util import *


def get_bad_samples(d_loss: Dict[str, np.array], k: int = 32, save: str = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    :param d_loss: The loss of each text sample in each dataset by a model, in iteration order
    :param k: top #samples to keep
    :return: A list of text samples with the respective loss that the model performs the worst on, sorted by performance
    :param save: Save the results to a directory path
    """
    d_out, split = dict(), 'test'
    for dnm, loss in d_loss.items():
        idxs_top = np.argpartition(loss, -k)[-k:]
        s_idxs_top = set(idxs_top)
        idxs_top = np.sort(idxs_top)  # increasing order of index to align with iteration order
        # idxs_idxs_top = np.argsort(loss[idxs_top])
        # idxs_top = idxs_top[].flip()  # sorted
        txts = []
        for i, t in enumerate(utcd.get_dataset(dnm, split).keys()):
            if i in s_idxs_top:
                txts.append(t)
        lst_txt_n_loss = [(t, float(loss[i])) for i, t in zip(idxs_top, txts)]
        lst_txt_n_loss = sorted(lst_txt_n_loss, key=lambda x: -x[1])  # sort by loss, descending
        d_out[dnm] = lst_txt_n_loss
    if save:
        fnm = os_join(save, f'{now(for_path=True)}, bad_samples.json')
        with open(fnm, 'w') as fl:
            json.dump(d_out, fl, indent=4)
    return d_out


if __name__ == '__main__':
    import pickle

    from icecream import ic

    model_dir_nm = os_join('binary-bert-rand-vanilla-old-shuffle-05.03.22', 'rand')
    path_eval = os_join(u.proj_path, u.model_dir, model_dir_nm, 'eval', 'in-domain, 05.09.22')
    with open(os_join(path_eval, 'eval_loss.pkl'), 'rb') as f:
        d = pickle.load(f)
    ic(get_bad_samples(d, save=path_eval))
