"""
"Pretrain" binary BERT for aspect classification given text only

For now just train with linear CLS objective

TODO: consider +MLM?
"""

from os.path import join as os_join
from typing import List, Tuple, Dict, Any

import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, SchedulerType
)
from transformers.training_args import OptimizerNames
from datasets import Dataset, ClassLabel, load_metric

from stefutil import *
from zeroshot_encoder.util import *
import zeroshot_encoder.util.utcd as utcd_util
from zeroshot_encoder.preprocess import get_dataset as get_dset


MODEL_NAME = 'Pretrain Aspect BinBERT'
HF_MODEL_NAME = 'bert-base-uncased'


def get_dataset(**kwargs) -> Tuple[Dataset, Dataset]:
    """
    override text classification labels to be aspect labels
    """
    dnm = 'UTCD-in'  # concatenated 9 in-domain datasets in UTCD
    # perform preprocessing outside `get_dataset` as feature from the dataset is needed
    dset = get_dset(dnm, **kwargs)
    trn: Dataset = dset[0]
    tst: Dataset = dset[1]

    aspects: List[str] = sconfig('UTCD.aspects')
    aspect2id = {a: i for i, a in enumerate(aspects)}
    # get aspect based on dataset id
    feat: ClassLabel = trn.features['dataset_id']  # the same feature for test
    did2aspect_id = {i: aspect2id[sconfig(f'UTCD.datasets.{feat.int2str(i)}.aspect')] for i in range(feat.num_classes)}

    def map_fn(samples: Dict[str, List[Any]]):
        ret = tokenizer(samples['text'], padding='max_length', truncation=True)
        ret['labels'] = [did2aspect_id[asp] for asp in samples['dataset_id']]
        return ret
    rmv = ['dataset_id', 'text']
    trn = trn.map(map_fn, batched=True, remove_columns=rmv)
    tst = tst.map(map_fn, batched=True, remove_columns=rmv)
    return trn, tst


class MyTrainer(Trainer):
    """
    Override `compute_loss` for getting training stats
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Begin of added ==========================
        # if model.training:
        #     ic(outputs.keys())
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def get_train_args() -> TrainingArguments:
    debug = True
    if debug:
        args = dict(
            batch_size=4,
            learning_rate=1e-4,
            weight_decay=0,
            lr_scheduler_type=SchedulerType.CONSTANT,
            num_train_epochs=4
        )
    else:
        # TODO: Keep those the same as in other approaches?; See `zeroshot_encoder.bi_encoder.dual_bi_encoder.py`
        args = dict(
            learning_rate=2e-5,
            train_batch_size=16,
            eval_batch_size=64,
            weight_decay=1e-2,
            num_train_epochs=3,
            lr_scheduler_type=SchedulerType.COSINE,
        )
    if 'batch_size' in args:
        bsz = args.pop('batch_size')
        args['per_device_train_batch_size'] = bsz
        args['per_device_eval_batch_size'] = bsz
    args.update(dict(
        output_dir=os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, MODEL_NAME, now(for_path=True)),
        do_train=True, do_eval=True,
        evaluation_strategy='epoch',
        eval_accumulation_steps=128,  # Saves GPU memory
        warmup_ratio=1e-1,
        adam_epsilon=1e-6,
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        optim=OptimizerNames.ADAMW_TORCH,
    ))
    return TrainingArguments(**args)


acc = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return dict(acc=acc.compute(predictions=preds, references=labels)['accuracy'])


if __name__ == '__main__':
    import transformers
    from icecream import ic

    # seed = sconfig('random-seed')
    seed = 42
    transformers.set_seed(seed)

    logger = get_logger(MODEL_NAME)
    logger.info('Setting up training... ')

    n = 4096
    # n = None
    logger.info('Loading tokenizer & model... ')
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=len(sconfig('UTCD.aspects')))

    logger.info('Loading data... ')
    tr, ts = get_dataset(n_sample=n, shuffle_seed=seed)
    logger.info(f'Loaded {logi(len(tr))} training samples, {logi(len(ts))} eval samples')

    trainer = MyTrainer(
        model=mdl, args=get_train_args(), train_dataset=tr, eval_dataset=ts, compute_metrics=compute_metrics
    )
    logger.info('Launching Training... ')
    trainer.train()
