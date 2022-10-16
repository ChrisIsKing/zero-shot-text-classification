import os
from os.path import join as os_join

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


if __name__ == '__main__':
    seed = sconfig('random-seed')

    NORMALIZE_ASPECT = True
    mic(NORMALIZE_ASPECT)

    def train(resume: str = None):
        logger = get_logger(f'{MODEL_NAME} Train')
        logger.info('Setting up training... ')

        # n = 128
        n = None

        lr = 4e-5
        n_ep = 8

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
        dset_args = dict(dataset_name=dnm, tokenizer=tokenizer, n_sample=n, shuffle_seed=seed)
        if NORMALIZE_ASPECT:
            dset_args.update(dict(normalize_aspect=seed, splits=['train', 'eval', 'test']))
        dsets = get_explicit_dataset(**dset_args)
        tr, vl, ts = dsets['train'], dsets['eval'], dsets['test']
        logger.info(f'Loaded #example {pl.i({k: len(v) for k, v in dsets.items()})}')

        transformers.set_seed(seed)
        path = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), mode='explicit',
            sampling=None, normalize_aspect=NORMALIZE_ASPECT
        )
        train_args = dict(
            output_dir=path,
            learning_rate=lr,
            per_device_train_batch_size=4,  # to fit in memory, bsz 16 to keep same with Bin Bert pretraining
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available(),
            num_train_epochs=n_ep,
            dataloader_num_workers=4
        )
        if NORMALIZE_ASPECT:
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
    train()
    # dir_nm_ = '2022-06-19_13-13-54_Explicit-Pretrain-Aspect-NVIDIA-GPT2-gpt2-medium-explicit-aspect-norm'
    # ckpt_path = os_join(utcd_util.get_base_path(), u.proj_dir, u.model_dir, dir_nm_, 'checkpoint-31984')
    # mic(ckpt_path)
    # train(resume=ckpt_path)

    dir_nm_ = '2022-06-12_16-40-16_Explicit Pretrain Aspect NVIDIA-GPT2-gpt2-medium-explicit-aspect-norm'
    save_path_ = os_join(u.proj_path, u.model_dir, dir_nm_, 'trained')

    def fix_save_tokenizer():
        tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_NAME)
        tokenizer.add_special_tokens(special_tokens_dict=dict(
            pad_token=ZsGPT2Tokenizer.pad_token_, additional_special_tokens=[utcd_util.EOT_TOKEN]
        ))
        mic(tokenizer.get_added_vocab())
        mic(save_path_)
        # exit(1)
        tokenizer.save_pretrained(save_path_)
        mic(os.listdir(save_path_))
    # fix_save_tokenizer()

    def check_save_tokenizer():
        tokenizer = GPT2TokenizerFast.from_pretrained(save_path_)
        mic(tokenizer.get_added_vocab())
    # check_save_tokenizer()
