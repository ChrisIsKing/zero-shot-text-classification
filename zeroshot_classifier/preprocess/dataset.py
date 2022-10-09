import os
from os.path import join as os_join
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
    logger.info(f'Loading dataset {pl.i(dataset_name)}... ')
    if from_disk:
        path = os_join(utcd_util.get_output_base(), u.proj_dir, u.dset_dir, 'processed', dataset_name)
        dsets = datasets.load_from_disk(path)

        if normalize_aspect:  # TODO: ugly but works
            logger.info(f'Normalizing training data by #sample per aspect with {pl.i(normalize_aspect)}...')
            _data = load_data.get_datasets(load_data.in_domain_data_path, normalize_aspect=normalize_aspect)
            # apply #sample normalization to the training set
            id2nm = sconfig('UTCD.dataset_id2name')
            # cos the same text may appear in multiple datasets
            dsets['train'] = dsets['train'].filter(
                lambda example: example['text'] in _data[id2nm[example['dataset_id']]]['train'])
            n = len(dsets['train'])
            # sanity check, same # of pairs as in Chris' data loading
            assert n == sum(len(d['train']) for d in _data.values())
            logger.info(f'Remaining #train pairs: {pl.i(n)}')
    else:
        dsets = datasets.load_dataset(dataset_name)
    if isinstance(splits, str):
        splits = [splits]
    dsets = [dsets[s] for s in splits]
    num_proc = None
    n_cpu = os.cpu_count()
    if fast and n_cpu >= 2 and min(len(d) for d in dsets) > 4096:
        num_proc = n_cpu
        if not pbar:
            datasets.set_progress_bar_enabled(False)
    # ordering of filter, shuffle, then select determined for debugging
    if filter_func is not None:
        logger.info('Filtering...')
        dsets = [dset.filter(filter_func, num_proc=num_proc) for dset in dsets]
    if shuffle_seed:
        logger.info(f'Shuffling with seed {pl.i(shuffle_seed)}...')
        dsets = [dset.shuffle(seed=shuffle_seed) for dset in dsets]
    if n_sample is not None:
        logger.info(f'Selecting the first {pl.i(n_sample)} samples...')
        dsets = [d.select(range(min(n_sample, len(d)))) for d in dsets]
    if map_func is not None:
        logger.info('Mapping...')
        if not isinstance(map_func, dict):
            map_func = {s: map_func for s in splits}
        dsets = [
            dset.map(map_func[split], batched=True, remove_columns=remove_columns, num_proc=num_proc)
            for dset, split in zip(dsets, splits)
        ]
    datasets.set_progress_bar_enabled(True)
    return dsets


class ExplicitMap:
    def __init__(self, tokenizer, dataset_name: str, dataset: List[Dataset]):
        self.tokenizer = tokenizer

        self.is_combined = 'UTCD' in dataset_name
        aspects: List[str] = sconfig('UTCD.aspects')
        aspect2id = {a: i for i, a in enumerate(aspects)}
        if self.is_combined:  # get aspect based on dataset id
            feat: ClassLabel = dataset[0].features['dataset_id']  # the same feature for both `train` and `test`
            n_dset = feat.num_classes
            self.did2aspect_id = {i: aspect2id[sconfig(f'UTCD.datasets.{feat.int2str(i)}.aspect')] for i in range(n_dset)}
        else:  # single dataset, the same aspect
            self.aspect_id = aspect2id[sconfig(f'UTCD.datasets.{dataset_name}.aspect')]

    def __call__(self, samples: Dict[str, List[Any]]):
        ret = self.tokenizer(samples['text'], padding='max_length', truncation=True)
        if self.is_combined:
            ret['labels'] = [self.did2aspect_id[asp] for asp in samples['dataset_id']]
        else:
            ret['labels'] = [self.aspect_id] * len(samples['text'])
        return ret


def get_explicit_dataset(
        dataset_name: str = 'UTCD-in', tokenizer: PreTrainedTokenizerBase = None, **kwargs
) -> List[Dataset]:
    """
    override text classification labels to be aspect labels
    """
    # perform preprocessing outside `get_dataset` as feature from the dataset is needed
    dsets = get_dataset(dataset_name, **kwargs)  # by split

    exp_map = ExplicitMap(tokenizer=tokenizer, dataset_name=dataset_name, dataset=dsets)

    rmv = ['text']
    if exp_map.is_combined:
        rmv.append('dataset_id')
    dsets = [dset.map(exp_map, batched=True, remove_columns=rmv, load_from_cache_file=False) for dset in dsets]
    return dsets
