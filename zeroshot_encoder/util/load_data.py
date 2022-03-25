import json
import gdown
import random
import spacy
import gzip
import csv
import os
import time
import itertools
from typing import Tuple, List
from collections import Counter
from tqdm import tqdm
from numpy import argmax, argmin
from os import listdir
from os.path import isfile, join, basename
from zipfile import ZipFile
from sentence_transformers.readers import InputExample
from sentence_transformers import util
from zeroshot_encoder.util import get_logger, logi, log_s, log_dict

in_domain_url = 'https://drive.google.com/uc?id=1V7IzdZ9HQbFUQz9NzBDjmqYBdPd9Yfe3'
out_of_domain_url = 'https://drive.google.com/uc?id=1nd32_UrFbgoCgH4bDtFFD_YFZhzcts3x'
dataset_path = './dataset'
in_domain_data_path = './dataset/in-domain'
out_of_domain_data_path = './dataset/out-of-domain'
nlp = spacy.load("en_core_web_md")
nlp.disable_pipes(['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])

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

def get_data(path):
    if not os.path.exists(path):
        download_data(path)
    paths = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('.json')]
    data = {}
    for path in paths:
        dataset_name = basename(path).split('.')[0]
        data[dataset_name] = json.load(open(path))
    return data

def download_data(path, file=None):
    if path == in_domain_data_path:
        file = './dataset/in-domain.zip'
        gdown.download(in_domain_url, file, quiet=False)
    elif path == out_of_domain_data_path:
        file = './dataset/out-domain.zip'
        gdown.download(out_of_domain_url, file, quiet=False)
    with ZipFile(file, "r") as zip:
        zip.extractall(dataset_path)
        zip.close()

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

def binary_cls_format(data, name=None, sampling='rand', train=True):
    examples = []
    if train:
        label_list = data['labels']
        example_list = [x for x in data['train'].keys()]

        if sampling == 'vect':
            label_vectors = {label: nlp(label) for label in label_list}
            start = time.time()
            vects = list(nlp.pipe(example_list, n_process=4, batch_size=128))
            print('Time Elapsed {} ms'.format((time.time() - start)*1000))
        
        print('Generating {} examples'.format(name))
        for i, (text, labels) in enumerate(tqdm(data['train'].items())):
            true_labels = labels
            other_labels = [label for label in label_list if label not in true_labels]
            
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

    else:
        for text, labels in data['test'].items():
           for label in labels:
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

        if sampling == 'vect':
            label_vectors = {label: nlp(label) for label in label_list}
            start = time.time()
            vects = list(nlp.pipe(example_list, n_process=4, batch_size=128))
            print('Time Elapsed {} ms'.format((time.time() - start)*1000))
        
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
                    examples.append(InputExample(texts=[text, nli_template(other_labels[argmax(scores)], data['aspect'])], label=0))
                    examples.append(InputExample(texts=[text, nli_template(other_labels[argmin(scores)], data['aspect'])], label=0))
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
        label_list = list(dict.fromkeys([example[1] for example in arr]))
        label_vectors = {label: nlp(label) for label in label_list}
        example_list = [x[0] for x in arr]

        if sampling == 'vect':
            start = time.time()
            vects = list(nlp.pipe(example_list, n_process=4, batch_size=128))
            print('Time Elapsed {} ms'.format((time.time() - start)*1000))
        
        # count instances
        count = Counter(example_list)
        has_multi_label = any((c > 1) for c in count.values())
        if has_multi_label:  # Potentially all valid labels for each text
            arr_ = sorted(arr)  # map from unique text to all possible labels
            txt2lbs = {k: set(lb for txt, lb in v) for k, v in itertools.groupby(arr_, key=lambda pair: pair[0])}
            logger = get_logger('Preprocess multi-label negative sampling')
            logger.info(f'Generating examples for dataset {logi(name)}, with labels {logi(label_list)}... ')
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
                        logger.warning(f'{log_s(warn_name, c="y", bold=True)}: # negative labels for text less '
                                       f'than {2}: {log_dict(text=txt, pos_labels=txt2lbs[txt], neg_labels=neg_pool)}')
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