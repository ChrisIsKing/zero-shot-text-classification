from os.path import join as os_join
from argparse import ArgumentParser

import torch
import transformers
from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_explicit_dataset
from zeroshot_classifier.models.gpt2 import MODEL_NAME as GPT2_MODEL_NAME, HF_MODEL_NAME, ZsGPT2Tokenizer
from zeroshot_classifier.models.explicit.explicit_v2 import *


MODEL_NAME = EXPLICIT_GPT2_MODEL_NAME
TRAIN_STRATEGY = 'explicit'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--normalize_aspect', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=8)

    return parser.parse_args()


if __name__ == '__main__':
    seed = sconfig('random-seed')

    def train(
            resume: str = None, normalize_aspect=True,
            learning_rate=4e-5, batch_size: int = 4, gradient_accumulation_steps: int = 8, epochs: int = 8,
            output_dir: str = None
    ):
        logger = get_logger(f'{MODEL_NAME} Train')
        logger.info('Setting up training... ')

        lr, bsz, gas, n_ep = learning_rate, batch_size, gradient_accumulation_steps, epochs

        logger.info('Loading tokenizer & model... ')
        tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_NAME)
        tokenizer.add_special_tokens(special_tokens_dict=dict(
            pad_token=ZsGPT2Tokenizer.pad_token_, additional_special_tokens=[utcd_util.EOT_TOKEN]
        ))
        config = transformers.GPT2Config.from_pretrained(HF_MODEL_NAME)
        config.pad_token_id = tokenizer.pad_token_id  # Needed for Seq CLS
        config.num_labels = len(sconfig('UTCD.aspects'))
        model = GPT2ForSequenceClassification.from_pretrained(HF_MODEL_NAME, config=config)
        # Include `end-of-turn` token for sgd, cannot set `eos` for '<|endoftext|>' already defined in GPT2
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
            per_device_train_batch_size=bsz,  # to fit in memory, bsz 32 to keep same with Bin Bert pretraining
            per_device_eval_batch_size=bsz,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
            num_train_epochs=n_ep,
            dataloader_num_workers=4
        )
        if normalize_aspect:
            train_args.update(dict(
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False
            ))
        train_args = get_train_args(model_name=GPT2_MODEL_NAME, **train_args)
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

    def command_prompt():
        args = parse_args()
        train(**vars(args))
    command_prompt()
