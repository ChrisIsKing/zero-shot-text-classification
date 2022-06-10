import os
from os.path import join as os_join

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_explicit_dataset
from zeroshot_classifier.models.binary_bert import HF_MODEL_NAME
from zeroshot_classifier.models.explicit.explicit_v2 import *


MODEL_NAME = EXPLICIT_BERT_MODEL_NAME
TRAIN_STRATEGY = 'explicit'


if __name__ == '__main__':
    import transformers

    seed = sconfig('random-seed')

    NORMALIZE_ASPECT = True
    mic(NORMALIZE_ASPECT)

    def train(resume: str = None):
        logger = get_logger(MODEL_NAME)
        logger.info('Setting up training... ')

        # n = 128
        n = None
        logger.info('Loading tokenizer & model... ')
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=len(sconfig('UTCD.aspects')))
        tokenizer.add_special_tokens(dict(eos_token=utcd_util.EOT_TOKEN))  # end-of-turn for SGD
        mdl.resize_token_embeddings(len(tokenizer))

        logger.info('Loading data... ')
        dnm = 'UTCD-in'  # concatenated 9 in-domain datasets in UTCD
        dset_args = dict(dataset_name=dnm, tokenizer=tokenizer, n_sample=n, shuffle_seed=seed)
        if NORMALIZE_ASPECT:
            dset_args['normalize_aspect'] = seed
        tr, vl = get_explicit_dataset(**dset_args)
        logger.info(f'Loaded {logi(len(tr))} training samples, {logi(len(vl))} eval samples')
        transformers.set_seed(seed)

        sanity_check_speed = False
        if sanity_check_speed:
            import torch.nn as nn
            from transformers import get_cosine_schedule_with_warmup
            from tqdm.auto import tqdm
            # bsz = 16
            # bsz = 8
            bsz = 32
            lr, decay = 2e-5, 1e-2
            num_train_epoch = 3
            mic(bsz, lr, decay, num_train_epoch)

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
                    model_name=MODEL_NAME,
                    # save_strategy='no'
                )
            else:
                with_tqdm = True
                args = get_train_args(
                    model_name=MODEL_NAME,
                    per_device_eval_batch_size=128
                )
            trainer_args = dict(
                model=mdl, args=args, train_dataset=tr, eval_dataset=vl, compute_metrics=compute_metrics
            )
            trainer_ = ExplicitTrainer(name=f'{MODEL_NAME} Training', with_tqdm=with_tqdm, **trainer_args)
            logger.info('Launching Training... ')
            if resume:
                trainer_.train(resume_from_checkpoint=resume)
            else:
                trainer_.train()
            save_path = os_join(trainer_.args.log_output_dir, 'trained')
            trainer_.save_model(save_path)
            tokenizer.save_pretrained(save_path)
            mic(save_path)
            mic(os.listdir(save_path))
    train()
    # dir_nm_ = '2022-05-16_21-25-30/checkpoint-274088'
    # ckpt_path = os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, MODEL_NAME.replace(' ', '-'), dir_nm_)
    # train(resume_from_checkpoint=ckpt_path)

    def evaluate(domain: str = 'in', batch_size: int = 32):
        ca(dataset_domain=domain)

        # dir_nm = '2022-05-16_21-25-30/checkpoint-274088'
        dir_nm = '2022-05-19_23-33-50/checkpoint-411132'
        path = os_join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, MODEL_NAME.replace(' ', '-'), dir_nm)
        mic(path)
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
            vl = get_explicit_dataset(dataset_name=dnm, tokenizer=tokenizer, n_sample=n, splits='test')[0]
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
            mic(dnm, n_sample, acc__)
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
    # fix_save_tokenizer()
