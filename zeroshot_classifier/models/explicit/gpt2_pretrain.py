from os.path import join as os_join

from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_explicit_dataset
from zeroshot_classifier.models.gpt2 import HF_MODEL_NAME
from zeroshot_classifier.models.explicit.explicit_v2 import *


MODEL_NAME = EXPLICIT_GPT2_MODEL_NAME
TRAIN_STRATEGY = 'explicit'


if __name__ == '__main__':
    import os

    import transformers

    from stefutil import *

    seed = sconfig('random-seed')

    NORMALIZE_ASPECT = True
    mic(NORMALIZE_ASPECT)

    def train():
        logger = get_logger(f'{MODEL_NAME} Train')
        logger.info('Setting up training... ')

        n = 128
        # n = None
        logger.info('Loading tokenizer & model... ')

        tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_NAME)
        model = GPT2ForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        # Include `end-of-turn` token for sgd, cannot set `eos` for '<|endoftext|>' already defined in GPT2
        tokenizer.add_special_tokens(special_tokens_dict=dict(additional_special_tokens=[utcd_util.EOT_TOKEN]))
        model.resize_token_embeddings(len(tokenizer))

        logger.info('Loading data... ')
        dnm = 'UTCD-in'  # concatenated 9 in-domain datasets in UTCD
        dset_args = dict(dataset_name=dnm, tokenizer=tokenizer, n_sample=n, shuffle_seed=seed)
        if NORMALIZE_ASPECT:
            dset_args['normalize_aspect'] = seed
        tr, vl = get_explicit_dataset(**dset_args)
        logger.info(f'Loaded {logi(len(tr))} training samples, {logi(len(vl))} eval samples')

        transformers.set_seed(seed)
        train_args = get_train_args(model_name=MODEL_NAME)
        trainer_args = dict(
            model=model, args=train_args, train_dataset=tr, eval_dataset=vl, compute_metrics=compute_metrics
        )
        trainer = ExplicitTrainer(name=f'{MODEL_NAME} Train', with_tqdm=True, **trainer_args)
        logger.info('Launching Training... ')
        trainer.train()

        save_path = os_join(trainer.args.log_output_dir, 'trained')
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        mic(save_path)
        mic(os.listdir(save_path))
    train()
