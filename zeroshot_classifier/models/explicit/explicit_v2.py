"""
Pretraining for 2-stage explicit training

Given text, predict aspect with linear classification head
    Binary BERT & Bi-Encoder all pretrained via BERT
    GPT2-NVIDIA pretrained with GPT2

Pretrained weights loaded for finetuning
"""

from os.path import join as os_join

from transformers import TrainingArguments, SchedulerType
from transformers.training_args import OptimizerNames

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
import zeroshot_classifier.models.binary_bert


__all__ = ['EXPLICIT_BERT_MODEL_NAME', 'EXPLICIT_GPT2_MODEL_NAME', 'get_train_args']


_bert_md_nm = zeroshot_classifier.models.binary_bert.MODEL_NAME
_gpt2_md_nm = zeroshot_classifier.models.gpt2.MODEL_NAME
EXPLICIT_BERT_MODEL_NAME = f'Aspect Pretrain {_bert_md_nm}'
EXPLICIT_GPT2_MODEL_NAME = f'Aspect Pretrain {_gpt2_md_nm}'


def get_train_args(model_name: str, dir_name: str = None, **kwargs) -> TrainingArguments:
    ca.check_mismatch('Model Name', model_name, [_bert_md_nm, _gpt2_md_nm])
    debug = False
    if debug:
        args = dict(
            batch_size=16,
            learning_rate=1e-4,
            weight_decay=0,
            lr_scheduler_type=SchedulerType.CONSTANT,
            num_train_epochs=4
        )
    else:
        # Keep the same as in Binary BERT vanilla training
        args = dict(
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            weight_decay=1e-2,
            num_train_epochs=3,
            lr_scheduler_type=SchedulerType.COSINE,
        )
    if 'batch_size' in args:
        bsz = args.pop('batch_size')
        args['per_device_train_batch_size'] = bsz
        args['per_device_eval_batch_size'] = bsz
    md_nm = model_name.replace(' ', '-')
    dir_nm = dir_name or f'{now(for_path=True)}_{md_nm}'
    args.update(dict(
        output_dir=os_join(utcd_util.get_base_path(), u.proj_dir, u.model_dir, dir_nm),
        do_train=True, do_eval=True,
        evaluation_strategy='epoch',
        eval_accumulation_steps=128,  # Saves GPU memory
        warmup_ratio=1e-1,
        adam_epsilon=1e-6,
        log_level='warning',
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        optim=OptimizerNames.ADAMW_TORCH,
        report_to='none'  # I have my own tensorboard logging
    ))
    args.update(kwargs)
    return TrainingArguments(**args)
