from os.path import join as os_join

from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_explicit_dataset
from zeroshot_classifier.models.gpt2 import MODEL_NAME as GPT2_MODEL_NAME, HF_MODEL_NAME, ZsGPT2Tokenizer
from zeroshot_classifier.models.explicit.explicit_v2 import *


MODEL_NAME = EXPLICIT_GPT2_MODEL_NAME
TRAIN_STRATEGY = 'explicit'


if __name__ == '__main__':
    import os

    import torch
    import transformers

    from stefutil import *

    seed = sconfig('random-seed')

    NORMALIZE_ASPECT = True
    mic(NORMALIZE_ASPECT)

    def train():
        logger = get_logger(f'{MODEL_NAME} Train')
        logger.info('Setting up training... ')

        # n = 128
        n = None
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
            dset_args['normalize_aspect'] = seed
        tr, vl = get_explicit_dataset(**dset_args)
        logger.info(f'Loaded {logi(len(tr))} training samples, {logi(len(vl))} eval samples')

        transformers.set_seed(seed)
        train_args = dict(
            per_device_train_batch_size=4,  # to fit in memory, bsz 16 to keep same with Bin Bert pretraining
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available(),
        )
        dir_nm = map_model_output_path(
            model_name=MODEL_NAME.replace(' ', '-'), output_path=HF_MODEL_NAME, mode='explicit',
            sampling=None, normalize_aspect=NORMALIZE_ASPECT
        )
        mic(dir_nm)
        train_args = get_train_args(model_name=GPT2_MODEL_NAME, dir_name=dir_nm, **train_args)
        trainer_args = dict(
            model=model, args=train_args, train_dataset=tr, eval_dataset=vl, compute_metrics=compute_metrics
        )
        trainer = ExplicitTrainer(name=f'{MODEL_NAME} Train', with_tqdm=True, **trainer_args)
        logger.info('Launching Training... ')
        trainer.train()

        save_path = os_join(trainer.log_output_dir, 'trained')
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        mic(save_path)
        mic(os.listdir(save_path))
    # train()

    dir_nm_ = '2022-06-12_16-40-16_Explicit Pretrain Aspect NVIDIA-GPT2-gpt2-medium-explicit-aspect-norm'
    save_path = os_join(u.proj_path, u.model_dir, dir_nm_, 'trained')

    def fix_save_tokenizer():
        tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_NAME)
        tokenizer.add_special_tokens(special_tokens_dict=dict(
            pad_token=ZsGPT2Tokenizer.pad_token_, additional_special_tokens=[utcd_util.EOT_TOKEN]
        ))
        mic(tokenizer.get_added_vocab())
        mic(save_path)
        # exit(1)
        tokenizer.save_pretrained(save_path)
        mic(os.listdir(save_path))
    # fix_save_tokenizer()

    def check_save_tokenizer():
        tokenizer = GPT2TokenizerFast.from_pretrained(save_path)
        mic(tokenizer.get_added_vocab())
    check_save_tokenizer()
