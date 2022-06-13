import os
from typing import List, Tuple, Dict, Callable, Union, Any

from transformers import PreTrainedTokenizerBase
import datasets
from datasets import Dataset, ClassLabel

from stefutil import *
from zeroshot_classifier.util import *
from zeroshot_classifier.util import load_data
import zeroshot_classifier.util.utcd as utcd_util


__all__ = ['get_dataset', 'get_explicit_dataset']


def get_dataset(
        dataset_name='ag_news', normalize_aspect: bool = False,
        map_func: Union[Dict[str, Callable], Callable] = None, filter_func: Callable = None,
        remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, shuffle_seed: int = None, fast=True, from_disk=True,
        splits: Union[str, List[str], Tuple[str]] = ('train', 'test'), pbar: bool = False
) -> List[Dataset]:
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
        if not pbar:
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


def get_explicit_dataset(
        dataset_name: str = 'UTCD-in', tokenizer: PreTrainedTokenizerBase = None, **kwargs
) -> List[Dataset]:
    """
    override text classification labels to be aspect labels
    """
    # perform preprocessing outside `get_dataset` as feature from the dataset is needed
    dsets = get_dataset(dataset_name, **kwargs)  # by split

    aspects: List[str] = sconfig('UTCD.aspects')
    aspect2id = {a: i for i, a in enumerate(aspects)}
    is_combined = 'UTCD' in dataset_name
    if is_combined:  # get aspect based on dataset id
        feat: ClassLabel = dsets[0].features['dataset_id']  # the same feature for both `train` and `test`
        n_dset = feat.num_classes
        did2aspect_id = {i: aspect2id[sconfig(f'UTCD.datasets.{feat.int2str(i)}.aspect')] for i in range(n_dset)}
    else:  # single dataset, the same aspect
        aspect_id = aspect2id[sconfig(f'UTCD.datasets.{dataset_name}.aspect')]

    def map_fn(samples: Dict[str, List[Any]]):
        ret = tokenizer(samples['text'], padding='max_length', truncation=True)
        if is_combined:
            ret['labels'] = [did2aspect_id[asp] for asp in samples['dataset_id']]
        else:
            ret['labels'] = [aspect_id] * len(samples['text'])
        return ret
    rmv = ['text']
    if is_combined:
        rmv.append('dataset_id')
    dsets = [dset.map(map_fn, batched=True, remove_columns=rmv, load_from_cache_file=False) for dset in dsets]
    return dsets
