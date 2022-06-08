import os
from typing import List, Tuple, Dict, Callable, Union

import datasets

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util import load_data
import zeroshot_classifier.util.utcd as utcd_util


def get_dataset(
        dataset_name='ag_news', normalize_aspect: bool = False,
        map_func: Union[Dict[str, Callable], Callable] = None, filter_func: Callable = None,
        remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, shuffle_seed: int = None, fast=True, from_disk=True,
        splits: Union[str, List[str], Tuple[str]] = ('train', 'test')
) -> List[datasets.Dataset]:
    logger = get_logger('Get Dataset')
    if from_disk:
        path = os.path.join(utcd_util.get_output_base(), u.proj_dir, u.dset_dir, 'processed', dataset_name)
        dsets = datasets.load_from_disk(path)

        if normalize_aspect:  # TODO: ugly but works
            logger.info(f'Normalizing training data by #sample per aspect with {logi(normalize_aspect)}...')
            _data = load_data.get_data(load_data.in_domain_data_path, normalize_aspect=normalize_aspect)
            # apply #sample normalization to the training set
            id2nm = sconfig('UTCD.dataset_id2name')
            # cos the same text may appear in multiple datasets
            dsets['train'] = dsets['train'].filter(
                lambda example: example['text'] in _data[id2nm[example['dataset_id']]]['train'])
            n = len(dsets['train'])
            # sanity check, same # of pairs as in Chris' data loading
            assert n == sum(len(d['train']) for d in _data.values())
            logger.info(f'Remaining #train pairs: {logi(n)}')
    else:
        dsets = datasets.load_dataset(dataset_name)
    if isinstance(splits, str):
        splits = [splits]
    dsets = [dsets[s] for s in splits]
    num_proc = None
    n_cpu = os.cpu_count()
    if fast and n_cpu >= 2:
        num_proc = n_cpu
        datasets.set_progress_bar_enabled(False)
    # ordering of filter, shuffle, then select determined for debugging
    if filter_func is not None:
        dsets = [dset.filter(filter_func, num_proc=num_proc) for dset in dsets]
    if shuffle_seed:
        dsets = [dset.shuffle(seed=shuffle_seed) for dset in dsets]
    if n_sample is not None:
        dsets = [d.select(range(min(n_sample, len(d)))) for d in dsets]
    if map_func is not None:
        if not isinstance(map_func, dict):
            map_func = {s: map_func for s in splits}
        dsets = [
            dset.map(map_func[split], batched=True, remove_columns=remove_columns, num_proc=num_proc)
            for dset, split in zip(dsets, splits)
        ]
    datasets.set_progress_bar_enabled(True)
    return dsets
