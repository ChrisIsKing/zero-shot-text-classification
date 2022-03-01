"""
Implementation of UPenn BERT with MNLI approach

[Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach]
(https://arxiv.org/abs/1909.00161)

Assumes model already pretrained on MNLI
implementing their fine-tuning approach, which serves as our continued pretraining
"""

from typing import Iterator

from transformers import BatchEncoding
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from transformers import Trainer

from zeroshot_encoder.util import *
from zeroshot_encoder.preprocess import *



class ZsBertTokenizer(BertTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cache: datasets.ClassLabel = None
        tmpls = config('baselines.bert-mnli.templates')
        self.dnm2tpl = {dnm: tmpls[d['aspect']] for dnm, d in config('UTCD.datasets').items()}
        ic(self.dnm2tpl)

    def _call_paren(self, *args, **kwargs) -> Dict:
        # ic(args, kwargs)
        # ic(super().__call__)
        # ic(super().__call__(
        #     text='what expression would i use to say i love you if i were an italian',
        # ), **kwargs)
        # exit(1)
        return super().__call__(*args, **kwargs)

    def __call__(self, samples: Dict[str, List[Union[str, int]]], **kwargs):
        """
        :param sample: Batched samples with keys `text`, `label`, `dataset_id`
            Intended to use with `Dataset.map`
        """
        max_length = kwargs.get('max_length', None)
        is_batched = isinstance(samples['label'], (tuple, list))
        if max_length is None:
            max_length = self.model_max_length
        # ic(self.model_max_length)
        # ic(list(samples), len(samples['label']))
        # ic(list(samples))

        def call_single(dataset_id, text, label) -> Iterator[Dict]:
            ic('in call single')
            # TODO: Which **split** affects the set of labels to generate
            # Convert multi-class labels to binary labels: 0 => `non-entailment`, 1=> `entailment`
            if self.cache is None:
                self.cache = datasets.load_from_disk(
                    os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', 'UTCD')
                )['train'].features['label']  # TODO: assume `train` split

            dset_nm = config('UTCD.dataset_id2name')[dataset_id]
            descs = config(f'UTCD.datasets.{dset_nm}.labels.train')
            # ic(descs)
            ic(dataset_id, text, label)
            label = descs.index(self.cache.int2str(label))  # Global UTCD label to local dataset ordinal label
            ic(label)
            n_lb = len(descs)
            # for i_lb, desc in enumerate(descs):
            #     hypo = self.dnm2tpl[dset_nm].format(desc)
            #     ic(hypo)
            #     if i_lb == label:
            #         # Truncate the premise/text, not the hypothesis/label
            #         yield self._call_paren(text, hypo, padding='max_length', truncation='only_first') | dict(label=1)
            ret = self._call_paren(
                [text] * n_lb,
                [self.dnm2tpl[dset_nm].format(desc) for desc in descs],
                padding='max_length', truncation='only_first') | dict(label=[int(i==label) for i in range(n_lb)])
            ic(list(ret))
            ic(ret['label'])
            exit(1)

        if is_batched:
            ds = sum((list(call_single(d_id, txt, lb)) for d_id, txt, lb in zip(
                *[samples[k] for k in ['dataset_id', 'text', 'label']]
            )), start=[])
        else:
            ds = list(call_single(0, *[samples[k] for k in ['dataset_id', 'text', 'label']]))
        return BatchEncoding({k: [d[k] for d in ds] for k in ds[0]})  # Stack all the ids


def get_all_setup(
        model_name, dataset_name: str = 'UTCD',
        n_sample=None, random_seed=None, do_eval=True, custom_logging=True
) -> Tuple[BertForSequenceClassification, BertTokenizerFast, datasets.Dataset, datasets.Dataset, Trainer]:
    assert dataset_name == 'UTCD'
    path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, 'bert', 'mnli-pretrained')
    model_ = BertForSequenceClassification.from_pretrained(path)
    assert model_.num_labels == 2  # Binary classification of entailment

    model_name_ = 'bert-base-uncased'
    tokenizer_ = ZsBertTokenizer.from_pretrained(  # For we load from disk, field missing
        path, model_max_length=BertTokenizerFast.max_model_input_sizes[model_name_]
    )
    tr, vl = get_dset(
        'UTCD', map_func=tokenizer_, n_sample=n_sample, remove_columns=['label', 'text'], random_seed=random_seed,
        fast='debug' not in model_name
    )
    return model_, tokenizer_, tr, vl,


if __name__ == '__main__':
    from icecream import ic

    seed = config('random-seed')
    n = 1024

    nm = 'debug'
    # nm = 'model'

    get_all_setup(
        model_name=nm, dataset_name='UTCD',
        do_eval=False, custom_logging=True, n_sample=n, random_seed=seed
    )

