from os.path import join as os_join
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_explicit_dataset
from zeroshot_classifier.models.binary_bert import MODEL_NAME as BERT_MODEL_NAME, HF_MODEL_NAME
from zeroshot_classifier.models.explicit.explicit_v2 import *


MODEL_NAME = EXPLICIT_BERT_MODEL_NAME
TRAIN_STRATEGY = 'explicit'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--normalize_aspect', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)

    return parser.parse_args()


if __name__ == '__main__':
    import transformers

    seed = sconfig('random-seed')

    def train(
            resume: str = None, normalize_aspect=True, learning_rate: float = 2e-5, batch_size: int = 32,
            epochs: int = 8,output_dir: str = None
    ):
        logger = get_logger(f'{MODEL_NAME} Train')
        logger.info('Setting up training... ')

        lr, bsz, n_ep = learning_rate, batch_size, epochs

        logger.info('Loading tokenizer & model... ')
        tokenizer = BertTokenizerFast.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=len(sconfig('UTCD.aspects')))
        tokenizer.add_special_tokens(dict(eos_token=utcd_util.EOT_TOKEN))  # end-of-turn for SGD
        model.resize_token_embeddings(len(tokenizer))

        logger.info('Loading data... ')
        dnm = 'UTCD-in'  # concatenated 9 in-domain datasets in UTCD
        dset_args = dict(dataset_name=dnm, tokenizer=tokenizer, shuffle_seed=seed)
        if normalize_aspect:
            dset_args.update(dict(normalize_aspect=seed, splits=['train', 'eval', 'test']))
        dsets = get_explicit_dataset(**dset_args)
        tr, vl, ts = dsets['train'], dsets['eval'], dsets['test']
        logger.info(f'Loaded #example {pl.i({k: len(v) for k, v in dsets.items()})}')
        transformers.set_seed(seed)

        path = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), mode='explicit',
            sampling=None, normalize_aspect=normalize_aspect, output_dir=output_dir
        )
        train_args = dict(
            output_dir=path,
            learning_rate=lr,
            per_device_train_batch_size=bsz,
            per_device_eval_batch_size=bsz,
            num_train_epochs=n_ep,
            dataloader_num_workers=4
        )
        if normalize_aspect:
            train_args.update(dict(
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False
            ))
        train_args = get_train_args(model_name=BERT_MODEL_NAME, **train_args)
        trainer_args = dict(
            model=model, args=train_args, train_dataset=tr, eval_dataset=vl, compute_metrics=compute_metrics
        )
        trainer = ExplicitTrainer(name=f'{MODEL_NAME} Train', with_tqdm=True, **trainer_args)
        logger.info('Launching Training... ')
        if resume:
            trainer.train(resume_from_checkpoint=resume)
        else:
            trainer.train()
        save_path = os_join(trainer.args.output_dir, 'trained')
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f'Tokenizer & Model saved to {pl.i(save_path)}')
    # train()

    def evaluate(domain: str = 'in', batch_size: int = 32):
        ca(dataset_domain=domain)

        dir_nm = '2022-05-19_23-33-50/checkpoint-411132'
        path = os_join(utcd_util.get_base_path(), u.proj_dir, u.model_dir, MODEL_NAME.replace(' ', '-'), dir_nm)
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        dnms = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == domain]

        def collate_fn(batch):  # as in speed sanity check
            ret = {k: torch.stack([torch.tensor(b[k]) for b in batch]) for k in batch[0] if k != 'labels'}
            ret['labels'] = torch.tensor([b['labels'] for b in batch])
            return ret

        for dnm in dnms:
            vl = get_explicit_dataset(dataset_name=dnm, tokenizer=tokenizer, splits='test')[0]
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

    def command_prompt():
        args = parse_args()
        train(**vars(args))
    command_prompt()


