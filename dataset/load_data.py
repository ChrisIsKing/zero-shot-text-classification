import json
import gdown
import random
from os import listdir
from os.path import isfile, join, basename
from zipfile import ZipFile
from sentence_transformers.readers import InputExample

random.seed(42)

g_drive_url = 'https://drive.google.com/uc?id=1qISYYoQNGXtmGWrCsKoK-fBKt8MHXqR7'
data_path = './dataset'

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

def binary_cls_format(arr, train=True):
    examples = []
    if train:
        label_list = list(set([example[1] for example in arr]))
        for element in arr:
            if element[0] == '':
                print('empty string in class {}'.format(element[1]))
            examples.append(InputExample(texts=[element[1], element[0]], label=1))
            examples.append(InputExample(texts=[random.choice([label for label in label_list if label != element[1]]), element[0]], label=0))
    else:
        for element in arr:
           examples.append(InputExample(texts=[element[1], element[0]], label=1))
    return examples
    

