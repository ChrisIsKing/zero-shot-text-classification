"""
"Pretrain" binary BERT for aspect classification given text only

For now just train with linear CLS objective

TODO: consider +MLM?
"""

from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, ClassLabel

from zeroshot_encoder.util import *
from zeroshot_encoder.preprocess import get_dataset


MODEL_NAME = 'Aspect Pretrain BinBERT'
HF_MODEL_NAME = 'bert-base-uncased'


if __name__ == '__main__':
    import transformers
    from icecream import ic

    # seed = sconfig('random-seed')
    seed = 42
    transformers.set_seed(seed)

    n = 4096
    # n = None

    dnm = 'UTCD-in'  # concatenated 9 in-domain datasets in UTCD
    # perform preprocessing outside `get_dataset` as feature from the dataset is needed
    dset = get_dataset(dnm, n_sample=n)
    tr: Dataset = dset[0]
    ts: Dataset = dset[1]
    aspects: List[str] = sconfig('UTCD.aspects')
    aspect2id = {a: i for i, a in enumerate(aspects)}
    # tr_feat: ClassLabel = tr.features['labels'].feature
    # ts_feat: ClassLabel = ts.features['labels'].feature
    # label2aspect_id = dict(
    #     train={i: tr_feat.int2str(i) for i in range(tr_feat.num_classes)},
    #     test={i: ts_feat.int2str(i) for i in range(ts_feat.num_classes)}
    # )

    # get aspect based on dataset id
    feat: ClassLabel = tr.features['dataset_id']  # the same feature for test
    # ic(feat)
    # ic(aspect2id)
    did2aspect_id = {i: aspect2id[sconfig(f'UTCD.datasets.{feat.int2str(i)}.aspect')] for i in range(feat.num_classes)}
    # ic(did2aspect_id)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    def map_fn(samples: Dict[str, List[Any]]):
        ret = tokenizer(samples['text'], padding='max_length', truncation=True)
        ret['labels'] = [did2aspect_id[asp] for asp in samples['dataset_id']]
        return ret
        # return d_tok | dict(labels=d_asp)  # override text classification labels to be aspect labels
        # d_asp = label2aspect_id[split]
        # ic(type(samples), len(samples))
        # ic(samples[0])
        # ic(samples.keys())
        # exit(1)
        # return samples

    rmv = ['dataset_id', 'text']
    tr = tr.map(map_fn, batched=True, remove_columns=rmv)
    ts = ts.map(map_fn, batched=True, remove_columns=rmv)
    sps = tr[:2]
    ic(type(sps))
    ic(sps.keys(), sps['labels'])
