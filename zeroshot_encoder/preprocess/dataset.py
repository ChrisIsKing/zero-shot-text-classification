from typing import Callable

from datasets import load_dataset


from zeroshot_encoder.util import *


def get_dset(
        dataset_name='ag_news',
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True
) -> Tuple[datasets.Dataset, ...]:
    if dataset_name == 'UTCD':
        if 'clarity' in get_hostname():
            out_base = os.path.join('/data')
        else:
            out_base = PATH_BASE
        dset = datasets.load_from_disk(os.path.join(out_base, DIR_PROJ, DIR_DSET, 'processed', 'UTCD'))
    else:
        dset = load_dataset(dataset_name)
    tr, vl = dset['train'], dset['test']
    if n_sample is not None:
        tr = tr.select(range(n_sample))
        vl = vl.select(range(n_sample))
    if map_func is not None:
        num_proc = None
        n_cpu = os.cpu_count()
        if fast and n_cpu >= 2:
            num_proc = n_cpu // 2
            datasets.set_progress_bar_enabled(False)

        tr = tr.map(map_func, batched=True, remove_columns=remove_columns, num_proc=num_proc)
        vl = vl.map(map_func, batched=True, remove_columns=remove_columns, num_proc=num_proc)
        datasets.set_progress_bar_enabled(True)

        if 'dataset_name' in tr.column_names:
            tr = tr.remove_columns('dataset_name')  # TODO: Why is it added in the first place?
            vl = vl.remove_columns('dataset_name')
    if random_seed:
        tr, vl = tr.shuffle(seed=random_seed), vl.shuffle(seed=random_seed)
    return tr, vl
