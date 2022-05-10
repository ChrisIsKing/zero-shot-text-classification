"""
Get the text samples that models perform the worst on, and look for insights

See `zeroshot_encoder.baseline.binary_bert` in test mode
"""

from typing import List, Tuple, Dict

import numpy as np

from zeroshot_encoder.util import *


def get_bad_samples(d_loss: Dict[str, np.array], k: int = 32) -> Dict[str, List[Tuple[str, float]]]:
    """
    :param d_loss: The loss of each text sample in each dataset by a model, in iteration order
    :param k: top #samples to keep
    :return: A list of text samples with the respective loss that the model performs the worst on, sorted by performance
    """
    d_out, split = dict(), 'test'
    for dnm, loss in d_loss.items():
        idxs_top = np.argpartition(loss, -k)[-k:]
        # ic(loss.shape, idxs_top)
        s_idxs_top = set(idxs_top)
        idxs_top = np.sort(idxs_top)  # increasing order of index to align with iteration order
        # idxs_idxs_top = np.argsort(loss[idxs_top])
        # idxs_top = idxs_top[].flip()  # sorted
        txts = []
        for i, t in enumerate(utcd.get_dataset(dnm, split).keys()):
            if i in s_idxs_top:
                txts.append(t)
                # ic(i)
        lst_txt_n_loss = [(t, loss[i]) for i, t in zip(idxs_top, txts)]
        lst_txt_n_loss = sorted(lst_txt_n_loss, key=lambda x: -x[1])  # sort by loss, descending
        d_out[dnm] = lst_txt_n_loss
    return d_out


if __name__ == '__main__':
    import pickle
    from os.path import join as os_join

    from icecream import ic

    model_dir_nm = os_join('binary-bert-rand-vanilla-old-shuffle-05.03.22', 'rand')
    path = os_join(u.proj_path, u.model_dir, model_dir_nm, 'eval', 'in-domain, 05.09.22', 'eval_loss.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    ic(get_bad_samples(d))
