"""
"Pretrain" binary BERT for aspect classification given text only

For now just train with linear CLS objective

TODO: consider +MLM?
"""
from os.path import join as os_join
from typing import List, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, SchedulerType
)
from transformers.training_args import OptimizerNames
from datasets import Dataset, ClassLabel, load_metric
from tqdm.auto import tqdm


from stefutil import *
from zeroshot_encoder.util import *
import zeroshot_encoder.util.utcd as utcd_util
from zeroshot_encoder.preprocess import get_dataset as get_dset


MODEL_NAME = 'Pretrain Aspect BinBERT'
HF_MODEL_NAME = 'bert-base-uncased'


def get_dataset(
        dataset_name: str = 'UTCD-in', tokenizer: BertTokenizer = None, normalize_aspect: Union[bool, int] = False,
        **kwargs
) -> List[Dataset]:
    """
    override text classification labels to be aspect labels
    """
    # perform preprocessing outside `get_dataset` as feature from the dataset is needed
    dsets = get_dset(dataset_name, **kwargs)  # by split

    if normalize_aspect:  # TODO: ugly but works
        _data = load_data.get_data(load_data.in_domain_data_path, normalize_aspect=normalize_aspect)
        ic(sum(len(d['train']) for d in _data.values()))
        txts = set().union(*[d_dset['train'] for d_dset in _data.values()])
        ic(dsets, len(dsets))
        ic(len(txts))
        # apply #sample normalization to the training set
        id2nm = sconfig('UTCD.dataset_id2name')
        # dsets[0] = dsets[0].filter(lambda example: example['text'] in txts)
        # cos the same text may appear in multiple datasets
        dsets[0] = dsets[0].filter(lambda example: example['text'] in _data[id2nm[example['dataset_id']]]['train'])
        ic(len(dsets[0]))
        # exit(1)
    # trn: Dataset
    # tst: Dataset

    aspects: List[str] = sconfig('UTCD.aspects')
    aspect2id = {a: i for i, a in enumerate(aspects)}
    is_combined = 'UTCD' in dataset_name
    if is_combined:  # get aspect based on dataset id
        feat: ClassLabel = dsets[0].features['dataset_id']  # the same feature for both `train` and `test`
        n_dset = feat.num_classes
        did2aspect_id = {i: aspect2id[sconfig(f'UTCD.datasets.{feat.int2str(i)}.aspect')] for i in range(n_dset)}
    else:  # single dataset, the same aspect
        aspect_id = aspect2id[sconfig(f'UTCD.datasets.{dataset_name}.aspect')]

    def map_fn(samples: Dict[str, List[Any]]):
        ret = tokenizer(samples['text'], padding='max_length', truncation=True)
        if is_combined:
            ret['labels'] = [did2aspect_id[asp] for asp in samples['dataset_id']]
        else:
            ret['labels'] = [aspect_id] * len(samples['text'])
        return ret
    rmv = ['text']
    if is_combined:
        rmv.append('dataset_id')
    dsets = [dset.map(map_fn, batched=True, remove_columns=rmv, load_from_cache_file=False) for dset in dsets]
    # trn = trn.map(map_fn, batched=True, remove_columns=rmv)
    # tst = tst.map(map_fn, batched=True, remove_columns=rmv)
    # ic(dsets[0][:2])
    # for k, v in trn[:2].items():
    #     v = v[0]
    #     ic(k, type(v))
    #     if isinstance(v, list):
    #         ic(len(v))
    # exit(1)
    return dsets


def get_train_args(**kwargs) -> TrainingArguments:
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
        # TODO: Keep those the same as in other approaches?; See `zeroshot_encoder.bi_encoder.dual_bi_encoder.py`
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
    md_nm = MODEL_NAME.replace(' ', '-')
    args.update(dict(
        output_dir=os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, md_nm, now(for_path=True)),
        do_train=True, do_eval=True,
        evaluation_strategy='epoch',
        eval_accumulation_steps=128,  # Saves GPU memory
        warmup_ratio=1e-1,
        adam_epsilon=1e-6,
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        optim=OptimizerNames.ADAMW_TORCH,
        report_to='none'  # I have my own tensorboard logging
    ))
    args.update(kwargs)
    return TrainingArguments(**args)


acc = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return dict(acc=acc.compute(predictions=preds, references=labels)['accuracy'])


if __name__ == '__main__':
    import transformers
    from icecream import ic

    ic.lineWrapWidth = 512

    seed = sconfig('random-seed')

    def train(resume_from_checkpoint: str = None):
        logger = get_logger(MODEL_NAME)
        logger.info('Setting up training... ')

        # n = 128
        n = None
        logger.info('Loading tokenizer & model... ')
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=len(sconfig('UTCD.aspects')))
        # tokenizer.add_special_tokens(dict(eos_token='[eot]'))  # end-of-turn for SGD
        # mdl.resize_token_embeddings(len(tokenizer))

        logger.info('Loading data... ')
        dnm = 'UTCD-in'  # concatenated 9 in-domain datasets in UTCD
        tr, vl = get_dataset(
            dataset_name=dnm, tokenizer=tokenizer, normalize_aspect=seed, n_sample=n, shuffle_seed=seed
        )
        logger.info(f'Loaded {logi(len(tr))} training samples, {logi(len(vl))} eval samples')
        transformers.set_seed(seed)

        sanity_check_speed = False
        ic(sanity_check_speed)
        if sanity_check_speed:
            import torch.nn as nn
            from transformers import get_cosine_schedule_with_warmup
            from tqdm.auto import tqdm
            # bsz = 16
            # bsz = 8
            bsz = 32
            lr, decay = 2e-5, 1e-2
            num_train_epoch = 3
            ic(bsz, lr, decay, num_train_epoch)

            def collate_fn(batch):
                ret = {k: torch.stack([torch.tensor(b[k]) for b in batch]) for k in batch[0] if k != 'labels'}
                ret['labels'] = torch.tensor([b['labels'] for b in batch])
                return ret
                # ic(len(batch))
                # ic(type(batch))
                # exit(1)
                # return {k: torch.tensor(v) for k, v in batch.items()}

            dl = DataLoader(tr, batch_size=bsz, shuffle=True, pin_memory=True, collate_fn=collate_fn)
            optimizer = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=decay)
            warmup_ratio, n_step = 0.1, len(dl)
            args = dict(optimizer=optimizer, num_warmup_steps=round(n_step*warmup_ratio), num_training_steps=n_step)
            scheduler = get_cosine_schedule_with_warmup(**args)

            if torch.cuda.is_available():
                mdl.cuda()

            epoch, step = 0, 0
            for _ in range(num_train_epoch):
                epoch += 1
                mdl.train()  # cos at the end of each eval, evaluate
                with tqdm(dl, desc=f'Train {epoch}', unit='ba') as t_dl:
                    for inputs in t_dl:
                        step += 1
                        optimizer.zero_grad()

                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                            # ic(inputs.keys())
                            # for k, v in inputs.items():
                            #     ic(k, v, type(v))
                            # inputs = {k: torch.tensor(v) for k, v in inputs.items()}
                        outputs = mdl(**inputs)
                        loss, logits = outputs.loss, outputs.logits.detach()
                        # labels = inputs['labels'].detach()
                        loss_scalar = loss.detach().item()

                        loss.backward()
                        nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0, error_if_nonfinite=True)
                        optimizer.step()
                        scheduler.step()

                        lr = scheduler.get_last_lr()[0]
                        t_dl.set_postfix(loss=loss_scalar, lr=lr)
        else:
            # debug = True
            debug = False
            if debug:
                # with_tqdm = False
                with_tqdm = True
                args = get_train_args(
                    # save_strategy='no'
                )
            else:
                with_tqdm = True
                args = get_train_args(
                    per_device_eval_batch_size=128
                )
            trainer_args = dict(model=mdl, args=args, train_dataset=tr, eval_dataset=vl, compute_metrics=compute_metrics)
            trainer_ = ExplicitBinBertTrainer(name=f'{MODEL_NAME} Training', with_tqdm=with_tqdm, **trainer_args)
            logger.info('Launching Training... ')
            if resume_from_checkpoint:
                trainer_.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer_.train()
    # train()
    # dir_nm_ = '2022-05-16_21-25-30/checkpoint-274088'
    # ckpt_path = os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, MODEL_NAME.replace(' ', '-'), dir_nm_)
    # train(resume_from_checkpoint=ckpt_path)

    def evaluate(domain: str = 'in', batch_size: int = 32):
        ca(dataset_domain=domain)

        # dir_nm = '2022-05-16_21-25-30/checkpoint-274088'
        dir_nm = '2022-05-19_23-33-50/checkpoint-411132'
        path = os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, MODEL_NAME.replace(' ', '-'), dir_nm)
        ic(path)
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)  # TODO: should add eot token as in updated training
        # tokenizer = BertTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        dnms = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == domain]
        # n = 1024
        n = None

        def collate_fn(batch):  # as in speed sanity check
            ret = {k: torch.stack([torch.tensor(b[k]) for b in batch]) for k in batch[0] if k != 'labels'}
            ret['labels'] = torch.tensor([b['labels'] for b in batch])
            return ret

        for dnm in dnms:
            vl = get_dataset(dataset_name=dnm, tokenizer=tokenizer, n_sample=n, splits='test')[0]
            n_sample = len(vl)
            dl = DataLoader(vl, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
            lst_preds, lst_labels = [], []
            with tqdm(dl, desc=f'Eval {dnm}', unit='ba') as it:
                for inputs in it:
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    outputs = model(**inputs)
                    logits = outputs.logits.detach()
                    labels = inputs['labels'].detach()
                    preds = torch.argmax(logits, dim=-1)
                    acc_ = (preds == labels).float().mean().item()
                    it.set_postfix(acc=acc_)
                    lst_preds.append(preds)
                    lst_labels.append(labels)
            preds = torch.cat(lst_preds, dim=0)
            labels = torch.cat(lst_labels, dim=0)
            acc__ = (preds == labels).float().mean().item()
            ic(dnm, n_sample, acc__)
    # evaluate(domain='out', batch_size=32)

    def fix_save_tokenizer():
        """
        To adhere to CrossEncoder API, save Bert tokenizer to the same directory
        """
        # dir_nm = '2022-05-19_23-33-50/checkpoint-411132'
        dir_nm = '2022-06-03_17-02-19/checkpoint-23988'
        path = os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, MODEL_NAME.replace(' ', '-'), dir_nm)
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)  # TODO: should add eot token as in updated training
        tokenizer.save_pretrained(path)
    fix_save_tokenizer()
