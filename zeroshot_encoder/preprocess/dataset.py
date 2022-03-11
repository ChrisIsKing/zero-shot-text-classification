from typing import Callable

from datasets import load_dataset


from zeroshot_encoder.util import *


def get_dset(
        dataset_name='ag_news',
        d_map_func: Dict[str, Callable] = None, filter_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True, from_disk=True,
        splits: Union[str, List[str], Tuple[str]] = ('train', 'test')
) -> List[datasets.Dataset]:
    if from_disk:
        dset = datasets.load_from_disk(os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', dataset_name))
    else:
        dset = load_dataset(dataset_name)
    if isinstance(splits, str):
        splits = [splits]
    dsets = [dset[s] for s in splits]
    if n_sample is not None:
        dsets = [d.select(range(n_sample)) for d in dsets]
    num_proc = None
    n_cpu = os.cpu_count()
    if fast and n_cpu >= 2:
        num_proc = n_cpu
        datasets.set_progress_bar_enabled(False)
    if filter_func is not None:
        dsets = [dset.filter(filter_func, num_proc=num_proc) for dset in dsets]
    if d_map_func is not None:
        dsets = [
            dset.map(d_map_func[split], batched=True, remove_columns=remove_columns, num_proc=num_proc)
            for dset, split in zip(dsets, splits)
        ]
    datasets.set_progress_bar_enabled(True)
    # dsets = [dset.remove_columns('dataset_name') for dset in dsets]  # TODO: Why is it added in the first place?
    if random_seed:
        dsets = [dset.shuffle(seed=random_seed) for dset in dsets]
    return dsets
