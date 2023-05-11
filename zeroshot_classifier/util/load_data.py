import os
import csv
import json
import time
import gzip
import random
import itertools
from os import listdir
from os.path import isfile, join as os_join, basename
from zipfile import ZipFile
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Dict, Union

import numpy as np
from numpy import argmax, argmin
import spacy
from sentence_transformers.readers import InputExample
from sentence_transformers import util
import gdown
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util import *


__all__ = [
    'in_domain_url', 'out_of_domain_url', 'in_domain_data_path', 'out_of_domain_data_path',
    'Dataset', 'SplitDataset',
    'get_datasets', 'to_aspect_normalized_datasets',
    'nli_template', 'get_nli_data', 'binary_cls_format', 'nli_cls_format', 'encoder_cls_format', 'seq_cls_format',
    'binary_explicit_format'
]


logger = get_logger('Load Data')


in_domain_url = 'https://drive.google.com/uc?id=1V7IzdZ9HQbFUQz9NzBDjmqYBdPd9Yfe3'
out_of_domain_url = 'https://drive.google.com/uc?id=1nd32_UrFbgoCgH4bDtFFD_YFZhzcts3x'
dataset_path = './dataset'
in_domain_data_path = './dataset/in-domain'
out_of_domain_data_path = './dataset/out-of-domain'
ASPECT_NORM_DIRNM = 'aspect-normalized'


def _get_nlp():
    if not hasattr(_get_nlp, 'nlp'):
        _get_nlp.nlp = spacy.load("en_core_web_md")
        _get_nlp.nlp.disable_pipes(['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    return _get_nlp.nlp


category_map = {
    "ag_news": "category",
    "clinc_150": "intent",
    "dbpedia": "category",
    "emotion": "sentiment",
    "sentiment_tweets_2020": "sentiment",
    "go_emotion": "sentiment",
    "sgd": "intent",
    "slurp": "intent",
    "yahoo": "category",
    "amazon_polarity": "sentiment",
    "multi_eurlex": "category",
    "banking77": "intent",
    "consumer_finance": "category",
    "finance_sentiment": "sentiment",
    "nlu_evaluation": "intent",
    "patent": "category",
    "snips": "intent",
    "yelp": "sentiment",
}


Dataset = Dict[str, List[str]]  # text => labels associated with that text
SplitDataset = Dict[str, Union[Dataset, List[str], str]]  # train & test splits + metadata including labels & aspect


def get_datasets(
        domain: str = 'in', n_sample: int = None, normalize_aspect: Union[bool, int] = False,
        dataset_names: Union[str, List[str]] = None
) -> [str, SplitDataset]:
    """
    :param n_sample: If given, a random sample of the entire dataset is selected
        Intended for debugging
    :param normalize_aspect: If true, # of training samples for each aspect is normalized
        via subsampling datasets in the larger aspect
        If int given, used as seed for sampling
    :param domain: Needed for aspect normalization
        Intended for training directly on out-of-domain data, see `zeroshot_classifier/models/bert.py`
    :param dataset_names: If given, only load the specified datasets
    """
    domain_paths = in_domain_data_path if domain == 'in' else out_of_domain_data_path
    domain_paths = [d for d in domain_paths.split(os.sep) if d != '.']
    path = os_join(u.proj_path, *domain_paths)
    if not os.path.exists(path):
        logger.info(f'Downloading {pl.i(domain)} domain data from GDrive to {pl.i(path)}...')
        download_data(path)
    datasets = None
    _keys = {'train', 'test', 'aspect', 'labels'}
    if normalize_aspect:
        if isinstance(normalize_aspect, int):
            assert normalize_aspect == sconfig('random-seed')
        path = os_join(path, ASPECT_NORM_DIRNM)
        if not os.path.exists(path):
            datasets = save_aspect_normalized_datasets(domain=domain)
        _keys.add('eval')
    if not datasets:
        if dataset_names:
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
            dataset_names = [f'{dnm}.json' for dnm in dataset_names]
        else:
            dataset_names = listdir(path)
        paths = [os_join(path, f) for f in dataset_names if isfile(os_join(path, f)) and f.endswith('.json')]
        datasets = dict()
        it = tqdm(paths, desc='Loading JSON datasets')
        for path in it:
            dataset_name = basename(path).split('.')[0]
            it.set_postfix(dataset=pl.i(dataset_name))
            dset = json.load(open(path))
            dset: SplitDataset

            assert set(dset.keys()) == _keys  # sanity check
            datasets[dataset_name] = dset

    splits = ['train', 'eval', 'test'] if normalize_aspect else ['train', 'test']
    if n_sample:
        for dnm, dsets in datasets.items():
            for sp in splits:
                dset = dsets[sp]
                n = len(dset) if normalize_aspect else sconfig(f'UTCD.datasets.{dnm}.splits.{sp}.n_text')
                if n < n_sample:
                    break

                if normalize_aspect:
                    txts = list(dset.keys())
                    txts = random.sample(txts, n_sample)
                else:  # TODO: support eval set
                    txts = np.empty(n, dtype=object)
                    for i, t in enumerate(dset.keys()):
                        txts[i] = t
                    txts = np.random.permutation(txts)[:n_sample]
                dsets[sp] = {t: dset[t] for t in txts}
    counts = {dnm: {sp: len(dsets[sp]) for sp in splits} for dnm, dsets in datasets.items()}
    logger.info(f'Datasets loaded w/ {pl.i(counts)}')
    return datasets


def subsample_dataset(dataset: Dataset = None, n_src: int = None, n_tgt: int = None, seed: int = None) -> Dataset:
    """
    Sample texts from text-labels pairs to roughly `n_sample` in total, while maintaining class distribution
    """
    if n_src is None:
        n_src = sum(len(lbs) for lbs in dataset.values())
    assert n_tgt < n_src
    ratio = n_tgt / n_src
    d_log = {'#source': n_src, '#target': n_tgt, 'subsample-ratio': f'{round(ratio * 100, 3)}%'}
    logger.info(f'Subsampling dataset w/ {pl.i(d_log)}... ')

    cls2txt = defaultdict(set)
    for txt, lbs in dataset.items():
        for lb in lbs:  # the same text may be added to multiple classes & hence sampled multiple times, see below
            cls2txt[lb].add(txt)
    # so that seed ensures reproducibility; TODO: too slow?
    cls2txt = {cls: sorted(txts) for cls, txts in cls2txt.items()}
    cls2count = {cls: len(txts) for cls, txts in cls2txt.items()}
    # normalize by #pair instead of #text for keeping output #text close to `n_sample`
    cls2count = {cls: round(c * ratio) for cls, c in cls2count.items()}  # goal count for output
    ret = dict()
    if seed:
        random.seed(seed)
    for cls, c in cls2count.items():
        to_sample = c
        while to_sample > 0:
            txts = random.sample(cls2txt[cls], to_sample)
            for t in txts:
                if t not in ret:  # ensure no-duplication in # samples added, since multi-label
                    ret[t] = dataset[t]
                    to_sample -= 1
                    cls2txt[cls].remove(t)
    return ret


def to_aspect_normalized_datasets(
        data: Dict[str, SplitDataset], seed: int = None, domain: str = 'in'
) -> Dict[str, SplitDataset]:
    """
    Sample the `train` split of the 9 in-domain datasets so that each `aspect` contains same # of samples

    Maintain class distribution
    """
    if seed:
        random.seed(seed)
    aspect2n_txt = defaultdict(int)
    for dnm, d_dset in sconfig('UTCD.datasets').items():
        if d_dset['domain'] == domain:
            aspect2n_txt[d_dset['aspect']] += d_dset['splits']['train']['n_text']
    logger.info(f'Aspect distribution: {pl.i(aspect2n_txt)}')
    asp_min = min(aspect2n_txt, key=aspect2n_txt.get)
    logger.info(f'Normalizing each aspect to ~{pl.i(aspect2n_txt[asp_min])} samples... ')

    for dnm, d_dset in data.items():
        asp = sconfig(f'UTCD.datasets.{dnm}.aspect')
        if asp != asp_min:
            n_normed = sconfig(f'UTCD.datasets.{dnm}.splits.train.n_text') * aspect2n_txt[asp_min] / aspect2n_txt[asp]
            n_src = sconfig(f'UTCD.datasets.{dnm}.splits.train.n_pair')
            d_dset['train'] = subsample_dataset(dataset=d_dset['train'], n_src=n_src, n_tgt=round(n_normed))
    dnm2count = defaultdict(dict)
    for dnm, d_dset in data.items():
        dnm2count[sconfig(f'UTCD.datasets.{dnm}.aspect')][dnm] = len(d_dset['train'])
    logger.info(f'Dataset counts after normalization: {pl.fmt(dnm2count)}')
    return data


def dataset2train_eval_split(dataset: Dataset, eval_ratio: float = 0.1, seed: int = None) -> Dict[str, Dataset]:
    """
    Split training set into train & eval set, try to maintain class distribution in both sets
    """
    if seed:
        random.seed(seed)
    # just like in `to_aspect_normalized_datasets::normalize_single`; the same text may be added to multiple classes
    cls2txt: Dict[str, Set[str]] = defaultdict(set)
    for txt, lbs in dataset.items():
        for lb in lbs:
            cls2txt[lb].add(txt)
    tr, vl = dict(), dict()
    txts_added = set()  # so that the same text is not added to both train & eval set
    for cls, txts in cls2txt.items():
        n_eval = max(round(len(txts) * eval_ratio), 1)  # at least 1 sample in eval for each class
        txts_vl = random.sample(txts, n_eval)
        txts_tr = txts - set(txts_vl)

        for t in txts_tr:
            if t not in txts_added:
                tr[t] = dataset[t]
                txts_added.add(t)
        for t in txts_vl:
            if t not in txts_added:
                vl[t] = dataset[t]
                txts_added.add(t)
    assert len(tr) + len(vl) == len(dataset)  # sanity check

    def dset2meta(dset: Dataset):
        labels = set().union(*dataset.values())
        return {'#text': len(dset), '#label': len(labels), 'labels': list(labels)}
    logger.info(f'Training set after split: {pl.i(dset2meta(tr))}')
    logger.info(f'Eval set after split: {pl.i(dset2meta(vl))}')
    return dict(train=tr, eval=vl)


def save_aspect_normalized_datasets(domain: str = 'in'):
    seed = sconfig('random-seed')
    path = in_domain_data_path if domain == 'in' else out_of_domain_data_path
    dsets = get_datasets(domain=domain)
    dsets = to_aspect_normalized_datasets(dsets, seed=seed, domain=domain)

    out_path = os.path.join(path, ASPECT_NORM_DIRNM)
    os.makedirs(out_path, exist_ok=True)

    ret = dict()
    for dnm, dsets_ in dsets.items():
        dsets__ = dataset2train_eval_split(dsets_.pop('train'), seed=seed)
        dsets__.update(dsets_)
        mic(dsets__.keys())
        path_out = os.path.join(out_path, f'{dnm}.json')
        logger.info(f'Saving normalized {pl.i(dnm)} dataset to {pl.i(path_out)}... ')
        with open(path, 'w') as f:
            json.dump(dsets__, f)
        ret[dnm] = dsets__
    return ret


def download_data(path):
    if 'in-domain' in path:
        file = './dataset/in-domain.zip'
        url = in_domain_url
    else:
        assert 'out-of-domain' in path
        file = './dataset/out-domain.zip'
        url = out_of_domain_url
    fl_paths = [d for d in file.split(os.sep) if d != '.']
    fl_path = os.path.join(u.proj_path, *fl_paths)
    os.makedirs(os.path.dirname(fl_path), exist_ok=True)
    gdown.download(url, fl_path, quiet=False)

    with ZipFile(file, "r") as zfl:
        dset_paths = [d for d in dataset_path.split(os.sep) if d != '.']
        path = os_join(u.proj_path, *dset_paths)
        os.makedirs(path, exist_ok=True)
        zfl.extractall(path)
        zfl.close()


def get_nli_data():
    nli_dataset_path = 'dataset/AllNLI.tsv.gz'

    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    train_samples = []
    dev_samples = []
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            label_id = label2int[row['label']]
            if row['split'] == 'train':
                train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
            else:
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    return train_samples, dev_samples


def binary_cls_format(
        dataset: SplitDataset = None, sampling='rand', split: str = 'train', mode='vanilla'
):
    ca.check_mismatch('Data Negative Sampling', sampling, ['rand', 'vect'])
    examples = []
    aspect = dataset['aspect']
    if split in ['train', 'eval']:
        aspect_token, sep_token = None, None
        label_un_modified = mode != 'implicit'
        ca(training_strategy=mode)
        if mode in ['vanilla', 'implicit-on-text-encode-aspect', 'implicit-on-text-encode-sep', 'explicit']:
            label_list = dataset['labels']
            if mode == 'implicit-on-text-encode-aspect':
                aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')[aspect]
            elif mode == 'implicit-on-text-encode-sep':
                sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')
        elif mode == 'implicit':
            label_list = ['{} {}'.format(label, dataset['aspect']) for label in dataset['labels']]
        else:
            raise NotImplementedError(f'{pl.i(mode)} not supported yet')

        example_list = [x for x in dataset[split].keys()]

        vects, label_vectors = None, None
        if sampling == 'vect':
            nlp = _get_nlp()
            label_vectors = {label: nlp(label) for label in label_list}
            start = time.time()
            vects = list(nlp.pipe(example_list, n_process=4, batch_size=128))
            print('Time Elapsed {} ms'.format((time.time() - start) * 1000))

        for i, (text, labels) in enumerate(dataset[split].items()):
            if label_un_modified:
                true_labels = labels
            else:
                true_labels = ['{} {}'.format(label, dataset['aspect']) for label in labels]
            other_labels = [label for label in label_list if label not in true_labels]

            if mode == 'implicit-on-text-encode-aspect':
                text = f'{aspect_token} {text}'
            elif mode == 'implicit-on-text-encode-sep':
                text = f'{aspect} {sep_token} {text}'

            # Generate label for true example
            for label in true_labels:
                examples.append(InputExample(texts=[text, label], label=1))

            # Generate sample based on sampling strategy
            if sampling == 'rand':
                random.seed(i)
                if len(other_labels) >= 2:
                    random_label = random.sample(other_labels, k=2)
                    # As expected by sentence-transformer::CrossEncoder
                    examples.append(InputExample(texts=[text, random_label[0]], label=float(0)))
                    examples.append(InputExample(texts=[text, random_label[1]], label=float(0)))
                elif len(other_labels) > 0:
                    examples.append(InputExample(texts=[text, other_labels[0]], label=float(0)))

            elif sampling == 'vect':
                if len(other_labels) >= 2:
                    text_vector = vects[i]
                    other_label_vectors = [label_vectors[label] for label in other_labels]
                    scores = [text_vector.similarity(vector) for vector in other_label_vectors]
                    examples.append(InputExample(texts=[text, other_labels[argmax(scores)]], label=float(0)))
                    examples.append(InputExample(texts=[text, other_labels[argmin(scores)]], label=float(0)))
                elif len(other_labels) > 0:
                    examples.append(InputExample(texts=[text, other_labels[0]], label=float(0)))

    else:  # test split
        aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')[aspect]
        sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')
        for text, labels in dataset['test'].items():
            for label in labels:
                if mode in ['vanilla', 'explicit']:
                    pass
                elif mode == 'implicit':
                    label = '{} {}'.format(label, aspect)
                elif mode == 'implicit-on-text-encode-aspect':
                    text = f'{aspect_token} {text}'
                elif mode == 'implicit-on-text-encode-sep':
                    text = f'{aspect} {sep_token} {text}'
                else:
                    raise NotImplementedError(f'{pl.i(mode)} not supported yet')
                examples.append(InputExample(texts=[text, label], label=1))
    return examples


def nli_template(label, category):
    if category == 'topic':
        return 'This text belongs to the topic of {}'.format(label)
    elif category == 'intent':
        return 'This text expresses the intent of {}'.format(label)
    elif category == 'sentiment':
        return 'This text expresses a {} sentiment'.format(label)


def nli_cls_format(data, name=None, sampling='rand', train=True):
    examples = []
    if train:
        label_list = data['labels']
        example_list = [x for x in data['train'].keys()]

        vects, label_vectors = None, None
        if sampling == 'vect':
            nlp = _get_nlp()
            label_vectors = {label: nlp(label) for label in label_list}
            start = time.time()
            vects = list(nlp.pipe(example_list, n_process=4, batch_size=128))
            print('Time Elapsed {} ms'.format((time.time() - start) * 1000))

        print('Generating {} examples'.format(name))
        for i, (text, labels) in enumerate(tqdm(data['train'].items())):
            true_labels = labels
            other_labels = [label for label in label_list if label not in true_labels]

            # Generate label for true example
            for label in true_labels:
                examples.append(InputExample(texts=[text, nli_template(label, data['aspect'])], label=1))

            # Generate sample based on sampling strategy
            if sampling == 'rand':
                random.seed(i)
                if len(other_labels) >= 2:
                    random_label = random.sample(other_labels, k=2)
                    examples.append(InputExample(texts=[text, nli_template(random_label[0], data['aspect'])], label=0))
                    examples.append(InputExample(texts=[text, nli_template(random_label[1], data['aspect'])], label=0))
                elif len(other_labels) > 0:
                    examples.append(InputExample(texts=[text, nli_template(other_labels[0], data['aspect'])], label=0))

            elif sampling == 'vect':
                if len(other_labels) >= 2:
                    text_vector = vects[i]
                    other_label_vectors = [label_vectors[label] for label in other_labels]
                    scores = [text_vector.similarity(vector) for vector in other_label_vectors]
                    examples.append(InputExample(
                        texts=[text, nli_template(other_labels[argmax(scores)], data['aspect'])], label=0))
                    examples.append(InputExample(
                        texts=[text, nli_template(other_labels[argmin(scores)], data['aspect'])], label=0))
                elif len(other_labels) > 0:
                    examples.append(InputExample(texts=[text, nli_template(other_labels[0])], label=0))

    else:
        for text, labels in data['test'].items():
            for label in labels:
                examples.append(InputExample(texts=[text, nli_template(label, data['aspect'])], label=1))
    return examples


WARN_NOT_ENOUGH_NEG_LABEL = 'Not Enough Negative Label'


def encoder_cls_format(
        arr: List[Tuple[str, str]], name=None, sampling='rand', train=True,
        neg_sample_for_multi=False, show_warnings=True
) -> List[InputExample]:
    """
    :param arr: List of dataset (text, descriptive label) pairs
    :param name: Dataset name
    :param sampling: Sampling approach, one of [`rand`, `vect`]
    :param train: If true, negative samples are generated
        Intended for training
    :param neg_sample_for_multi: If true, negative samples are added for each positive labels for a text
    :param show_warnings: If true, warning for missing negative labels are logged
    """
    examples = []
    if train:
        nlp = _get_nlp()
        label_list = list(dict.fromkeys([example[1] for example in arr]))
        label_vectors = {label: nlp(label) for label in label_list}
        example_list = [x[0] for x in arr]

        vects = None
        if sampling == 'vect':
            start = time.time()
            vects = list(nlp.pipe(example_list, n_process=4, batch_size=128))
            print('Time Elapsed {} ms'.format((time.time() - start) * 1000))

        # count instances
        count = Counter(example_list)
        has_multi_label = any((c > 1) for c in count.values())
        txt2lbs = None
        if has_multi_label:  # Potentially all valid labels for each text
            arr_ = sorted(arr)  # map from unique text to all possible labels
            txt2lbs = {k: set(lb for txt, lb in v) for k, v in itertools.groupby(arr_, key=lambda pair: pair[0])}
            logger.info(f'Generating examples for dataset {pl.i(name)}, with labels {pl.i(label_list)}... ')
        print('Generating {} examples'.format(name))
        for i, element in enumerate(tqdm(arr)):
            true_label = element[1]
            other_labels = [label for label in label_list if label != element[1]]

            # Generate label for true example
            examples.append(InputExample(texts=[true_label, element[0]], label=float(1)))

            # Generate sample based on sampling strategy
            if has_multi_label and neg_sample_for_multi:
                assert sampling == 'rand'  # TODO: vect not supported
                random.seed(i)
                txt = element[0]
                neg_pool = set(label_list) - txt2lbs[txt]

                def neg_sample2label(lb: str) -> InputExample:
                    return InputExample(texts=[lb, txt], label=float(0))

                if len(neg_pool) < 2:
                    # Ensures 2 negative labels are sampled, intended to work with existing Jaseci training code
                    warn_name = WARN_NOT_ENOUGH_NEG_LABEL
                    if len(neg_pool) == 0:
                        warn_name = f'{warn_name}, severe'
                        neg_label = 'dummy negative label'
                    else:
                        neg_label = list(neg_pool)[0]
                    examples.extend([neg_sample2label(neg_label), neg_sample2label(neg_label)])
                    if show_warnings:
                        logger.warning(f'{pl.s(warn_name, c="y", bold=True)}: # negative labels for text less '
                                       f'than {2}: {pl.i(text=txt, pos_labels=txt2lbs[txt], neg_labels=neg_pool)}')
                else:
                    examples.extend([neg_sample2label(neg_label) for neg_label in random.sample(neg_pool, k=2)])
            else:
                if sampling == 'rand' and count[element[0]] < 2:
                    random.seed(i)
                    random_label = random.sample(other_labels, k=2)
                    examples.append(InputExample(texts=[random_label[0], element[0]], label=float(0)))
                    examples.append(InputExample(texts=[random_label[1], element[0]], label=float(0)))
                elif sampling == 'vect' and count[element[0]] < 2:
                    text_vector = vects[i]
                    other_label_vectors = [label_vectors[label] for label in other_labels]
                    scores = [text_vector.similarity(vector) for vector in other_label_vectors]
                    examples.append(InputExample(texts=[other_labels[argmax(scores)], element[0]], label=float(0)))
                    examples.append(InputExample(texts=[other_labels[argmin(scores)], element[0]], label=float(0)))

    else:
        for element in arr:
            examples.append(InputExample(texts=[element[1], element[0]], label=float(1)))
    return examples


def seq_cls_format(data, all=False):
    train = []
    test = []
    label_map = {}

    if all:
        for dataset, item in data.items():
            for label in item['labels']:
                if label not in label_map:
                    label_map[label] = len(label_map)

            for k, v in item['train'].items():
                # loop through each true label
                for label in v:
                    train.append({'text': k, 'label': label_map[label], 'label_name': label})

            for k, v in item['test'].items():
                # loop through each true label
                for label in v:
                    test.append({'text': k, 'label': label_map[label], 'label_name': label})

    else:

        label_map = {k: i for i, k in enumerate(data['labels'])}

        for k, v in data['train'].items():
            # loop through each true label
            for label in v:
                train.append({'text': k, 'label': label_map[label], 'label_name': label})

        for k, v in data['test'].items():
            # loop through each true label
            for label in v:
                test.append({'text': k, 'label': label_map[label], 'label_name': label})
    return train, test, label_map


class ExplicitInputExample:
    def __init__(self, texts, label, aspect) -> None:
        self.texts = texts
        self.label = label
        self.aspect = aspect

    def __str__(self):
        return "<ExplicitInputExample> label: {}, text: {}".format(str(self.label), self.text)


def binary_explicit_format(dataset):
    aspect_map = {"sentiment": 0, "intent": 1, "topic": 2}

    train = []

    for name, data in dataset.items():
        aspect = data['aspect']
        label_list = data['labels']

        for i, (text, labels) in enumerate(tqdm(data['train'].items(), desc=name)):

            true_labels = labels
            other_labels = [label for label in label_list if label not in true_labels]

            # Generate label for true example
            for label in true_labels:
                train.append(ExplicitInputExample(texts=[text, label], label=1, aspect=aspect_map[aspect]))
                random.seed(i)
                if len(other_labels) >= 2:
                    random_label = random.sample(other_labels, k=2)
                    train.append(
                        ExplicitInputExample(texts=[text, random_label[0]], label=0, aspect=aspect_map[aspect]))
                    train.append(
                        ExplicitInputExample(texts=[text, random_label[1]], label=0, aspect=aspect_map[aspect]))
                elif len(other_labels) > 0:
                    train.append(
                        ExplicitInputExample(texts=[text, other_labels[0]], label=0, aspect=aspect_map[aspect]))
    return train


if __name__ == '__main__':
    mic.output_width = 512

    random.seed(sconfig('random-seed'))

    def check_sampling():
        data = get_datasets(in_domain_data_path)
        data = to_aspect_normalized_datasets(data)
        for dnm, d_dset in data.items():
            mic(dnm, len(d_dset['train']))
            c = Counter()
            for txt, lbs in d_dset['train'].items():
                c.update(lbs)
            mic(c, len(c))
    # check_sampling()

    save_aspect_normalized_datasets(domain='in')
    save_aspect_normalized_datasets(domain='out')
