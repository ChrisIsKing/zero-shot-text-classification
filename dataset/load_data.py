import json
import gdown
import random
import spacy
import gzip
import csv
import os
import time
from collections import Counter
from tqdm import tqdm
from numpy import argmax, argmin
from os import listdir
from os.path import isfile, join, basename
from zipfile import ZipFile
from sentence_transformers.readers import InputExample
from sentence_transformers import LoggingHandler, util

g_drive_url = 'https://drive.google.com/uc?id=1qISYYoQNGXtmGWrCsKoK-fBKt8MHXqR7'
data_path = './dataset'
nlp = spacy.load("en_core_web_md")
nlp.disable_pipes(['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])

def get_all_zero_data():
    if len(listdir(data_path)) <= 9:
        download_data()
    paths = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith('.json')]
    data = {"all": {"train": [], "test": []}}
    for path in paths:
        dataset_name = basename(path).split('.')[0]
        data[dataset_name] = json.load(open(path))
        data['all']["train"] += data[dataset_name]["train"]
        data['all']["test"] += data[dataset_name]["test"]
    return data

def download_data():
    path = '{}/data.zip'.format(data_path)
    gdown.download(g_drive_url, path, quiet=False)
    with ZipFile(path, "r") as zip:
        zip.extractall(data_path)
        zip.close()

def get_nli_data():
    nli_dataset_path = 'dataset/AllNLI.tsv.gz'

    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 0}
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

def binary_cls_format(arr, name=None, sampling='rand', train=True):
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
        print('Generating {} examples'.format(name))
        for i, element in enumerate(tqdm(arr)):
            true_label = element[1]
            other_labels = [label for label in label_list if label != element[1]]
            
            # Generate label for true example
            examples.append(InputExample(texts=[true_label, element[0]], label=1))

            # Generate sample based on sampling strategy
            if sampling == 'rand' and count[element[0]] < 2:
                random.seed(i)
                random_label = random.sample(other_labels, k=2)
                examples.append(InputExample(texts=[random_label[0], element[0]], label=0))
                examples.append(InputExample(texts=[random_label[1], element[0]], label=0))
            elif sampling == 'vect' and count[element[0]] < 2:
                text_vector = vects[i]
                other_label_vectors = [label_vectors[label] for label in other_labels]
                scores = [text_vector.similarity(vector) for vector in other_label_vectors]
                examples.append(InputExample(texts=[other_labels[argmax(scores)], element[0]], label=0))
                examples.append(InputExample(texts=[other_labels[argmin(scores)], element[0]], label=0))

    else:
        for element in arr:
           examples.append(InputExample(texts=[element[1], element[0]], label=1))
    return examples
    

