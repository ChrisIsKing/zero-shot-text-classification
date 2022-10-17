from torch import device
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
import numpy as np
from zeroshot_classifier.util.load_data import (
    get_datasets, binary_cls_format, in_domain_data_path, out_of_domain_data_path
)
from stefutil import *
from zeroshot_classifier.util import *

from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def filt(d, dom):
    return d['domain'] == dom

domain = 'out'
split = 'test'
data = get_datasets(domain=domain)
batch_size = 32
dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if filt(d_dset, domain)]

for dnm in dataset_names:  # loop through all datasets
    dset = data[dnm]
    pairs, aspect = dset[split], dset['aspect']
    d_dset = sconfig(f'UTCD.datasets.{dnm}.splits.{split}')
    label_options, multi_label = d_dset['labels'], d_dset['multi_label']
    n_options = len(label_options)
    label2id = {lbl: i for i, lbl in enumerate(label_options)}
    n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
    d_log = {'#text': n_txt, '#label': n_options, 'labels': label_options}

    print(f"Streaming batch_size={batch_size}")
    
    correct = 0
    for out in tqdm(classifier(list(pairs.keys()), label_options, batch_size=batch_size), total=len(pairs)):
        if out['labels'][np.argmax(out['scores'])] in pairs[out['sequence']]:
            correct += 1

    print(f"Accuracy: {correct/len(pairs)}")
  