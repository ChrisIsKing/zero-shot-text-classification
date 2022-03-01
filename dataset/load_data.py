import json
import gdown
import random
import spacy
from numpy import argmax, argmin
from os import listdir
from os.path import isfile, join, basename
from zipfile import ZipFile
from sentence_transformers.readers import InputExample

g_drive_url = 'https://drive.google.com/uc?id=1qISYYoQNGXtmGWrCsKoK-fBKt8MHXqR7'
data_path = './dataset'
nlp = spacy.load("en_core_web_md")

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

def binary_cls_format(arr, sampling='rand', train=True):
    examples = []
    if train:
        label_list = list(set([example[1] for example in arr]))
        for element in arr:
            true_label = element[1]
            other_labels = [label for label in label_list if label != element[1]]
            # Generate label for true example
            examples.append(InputExample(texts=[true_label, element[0]], label=1))

            # Generate sample based on sampling strategy
            if sampling == 'rand':
                random_label = random.sample(other_labels, k=2)
                examples.append(InputExample(texts=[random_label[0], element[0]], label=0))
                examples.append(InputExample(texts=[random_label[1], element[0]], label=0))
            elif sampling == 'vect':
                text_vector = nlp(element[0])
                other_label_vectors = [nlp(label) for label in other_labels]
                scores = [text_vector.similarity(vector) for vector in other_label_vectors]
                examples.append(InputExample(texts=[other_labels[argmax(scores)], element[0]], label=0))
                examples.append(InputExample(texts=[other_labels[argmin(scores)], element[0]], label=0))

    else:
        for element in arr:
           examples.append(InputExample(texts=[element[1], element[0]], label=1))
    return examples
    

