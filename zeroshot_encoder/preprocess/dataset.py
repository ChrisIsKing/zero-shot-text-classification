import os
from typing import List, Tuple, Dict, Callable, Union

import datasets

from zeroshot_encoder.util.data_path import PROJ_DIR, DSET_DIR
import zeroshot_encoder.util.utcd as utcd_util


def get_dataset(
        dataset_name='ag_news',
        map_func: Union[Dict[str, Callable], Callable] = None, filter_func: Callable = None,
        remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True, from_disk=True,
        splits: Union[str, List[str], Tuple[str]] = ('train', 'test')
) -> List[datasets.Dataset]:
    if from_disk:
        path = os.path.join(utcd_util.get_output_base(), PROJ_DIR, DSET_DIR, 'processed', dataset_name)
        dset = datasets.load_from_disk(path)
    else:
        dset = datasets.load_dataset(dataset_name)
    if isinstance(splits, str):
        splits = [splits]
    dsets = [dset[s] for s in splits]
    num_proc = None
    n_cpu = os.cpu_count()
    if fast and n_cpu >= 2:
        num_proc = n_cpu
        datasets.set_progress_bar_enabled(False)
    # ordering of filter, shuffle, then select determined for debugging
    if filter_func is not None:
        dsets = [dset.filter(filter_func, num_proc=num_proc) for dset in dsets]
    if random_seed:
        dsets = [dset.shuffle(seed=random_seed) for dset in dsets]
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
