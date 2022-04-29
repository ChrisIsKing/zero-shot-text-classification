"""
Implementation of NVIDIA-GPT2 approach.

[Zero-shot Text Classification With Generative Language Models](https://arxiv.org/abs/1912.10165)
"""

import os
import re
import math
import itertools
from typing import List, Tuple, Dict, Union, Any
from warnings import warn
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch import nn
from sklearn.metrics import classification_report
import transformers
from transformers import BatchEncoding
from transformers import AutoConfig
from transformers import GPT2TokenizerFast
from transformers import GPT2Model, GPT2LMHeadModel  # LMHead for CLM training
from transformers import Trainer, TrainingArguments, SchedulerType
from transformers import DataCollatorForLanguageModeling
from transformers.file_utils import ModelOutput
from transformers.training_args import OptimizerNames
import datasets
import matplotlib.pyplot as plt
import seaborn as sns

from stefutil import *
from zeroshot_encoder.util.train import MyEvalPrediction
import zeroshot_encoder.util.utcd as utcd_util
from zeroshot_encoder.preprocess import get_dataset


MODEL_NAME = 'gpt2-nvidia'


class ZsGPT2Tokenizer(GPT2TokenizerFast):
    """
    A wrapper around GPT2 tokenizer for 0-shot classification tokenizing
    """
    SPEC_TOKS = OrderedDict([
        ('pref_ques', '<|question|>'),  # Word embeddings
        ('pref_text', '<|text|>'),
        ('pref_answ', '<|answer|>'),
        ('sep_answ', '<|answer_sep|>'),  # Separation between answers if multiple answers
        ('type_ques', '[QUES]'),  # Type embeddings
        ('type_text', '[TEXT]'),
        ('type_answ', '[ANSW]')
    ])

    class Cache(dict):
        """
        Wrapper around caching dict, that loads metadata on corresponding dataset
        """
        def __init__(self, tokenizer: 'ZsGPT2Tokenizer'):
            super().__init__()
            self.tokenizer = tokenizer
            self.tpl_grouped = re.compile(rf'^(?P<dataset_name>.?)-label-grouped$')

        def __getitem__(self, key: Tuple[str, str]):
            """
            :param key: 2-tuple of (dataset_name, split)

            Needed cos huggingface may load cached dataset, internal cache is gone

            .. note:: works for local disk dataset only
            """
            dataset_name, split = key
            key = f'{dataset_name}-{split}'
            if key not in self:
                path = os.path.join(utcd_util.get_output_base(), PROJ_DIR, DSET_DIR, 'processed', dataset_name)
                dset = datasets.load_from_disk(path)[split]
                # See `zeroshot_encoder.util.util.py::process_utcd_dataset`
                feats = dset.features['labels'].feature
                n_cls = feats.num_classes
                assert feats.names == sconfig(f'UTCD.datasets.{dataset_name}.splits.{split}.labels')  # sanity check
                label2description: Dict[int, str] = {i: desc for i, desc in enumerate(feats.names)}  # label is index
                self[key] = dict(
                    n_classes=n_cls, label2description=label2description,
                    max_label_id_length=max(len(self.tokenizer._call_paren(lb)) for lb in feats.names)
                )
            return super().__getitem__(key)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pad token cannot be `self.eos_token`
        # cos otherwise `DataCollatorForLanguageModeling` would override normal eos tokens
        self.add_special_tokens(dict(
            pad_token='[PAD]', additional_special_tokens=list(ZsGPT2Tokenizer.SPEC_TOKS.values())
        ))

        self.templates = sconfig('baselines.gpt2-nvidia.templates')
        # Mapping from dataset name to label for non-UTCD cases
        # self.cache: Dict[Tuple[str, str], Dict] = ZsGPT2Tokenizer.Cache(self)
        self.cache = ZsGPT2Tokenizer.Cache(self)
        self.cache_utcd = None

        self.boq_token, self.bot_token, self.boa_token = (  # begin of (question, text, answer) tokens
            ZsGPT2Tokenizer.SPEC_TOKS[k] for k in ('pref_ques', 'pref_text', 'pref_answ')
        )  # Special tokens
        self.ques_sep_token = ZsGPT2Tokenizer.SPEC_TOKS['sep_answ']
        self.question_type_token, self.text_type_token, self.answer_type_token = (
            ZsGPT2Tokenizer.SPEC_TOKS[k] for k in ('type_ques', 'type_text', 'type_answ')
        )  # Type tokens

        self.warned_desc = set()  # Warning for each dataset happens once    @property

    @property
    def max_len_single_sentence(self) -> int:
        return self.model_max_length - 2 * 3  # 3 pairs of (special start token, eos token)

    def _call_paren(self, s: str, **kwargs) -> List[int]:
        return super().__call__(s, **kwargs)['input_ids']

    def enc_spec(self, tok: str) -> int:
        """
        Encode special tokens with sanity check
        """
        id_ = self.encode(tok)
        assert len(id_) == 1
        return id_[0]  # Intended for special tokens

    def __call__(
            self, samples: Dict[str, Union[List, str, int]],
            dataset_name: str = 'UTCD', split: str = 'train', mode: str = 'train', **kwargs
    ):
        """
        :param samples: Data sample(s) with keys [`dataset_name`, `label`, `text`]
            Each value an element or a list of elements
        :param split: One of [`train`, `test`]
            Shouldn't matter for UTCD datasets, see `process_utcd_dataset`
        :param mode: one of [`train`, `inference`, `stats`, `inference-debug],
            If `inference`, the answer part is not tokenized
                the text portion is truncated such that the label with largest # of ids may be generated;
                the batch is not padded
                    i.e. Intended for prediction, see `evaluate_trained`
            If `stats`, the entire sample is tokenized without truncation
        """
        modes = ['train', 'inference', 'stats', 'inference-sample']
        assert mode in modes, f'Unexpected mode: Expect one of {logi(modes)}, got {logi(mode)}'
        max_length = kwargs.get('max_length', None)
        is_batched = isinstance(samples['text'], (tuple, list))
        if max_length is None:
            max_length = self.model_max_length
        n_token = self.model_max_length  # Indented number of token positions as in the actual architecture

        ln = len(samples['text'])
        idxs_tpl = np.random.randint(len(self.templates), size=ln)

        def call_single(
                i, dataset_id: int = None, text: str = None, labels: List[int] = None, label_options: List[str] = None
        ):
            dset_nm: str = None if mode == 'inference-sample' else sconfig('UTCD.dataset_id2name')[dataset_id]
            if mode == 'inference-sample':
                assert label_options is not None
                n_cls = len(label_options)

                def lb_int2desc(lb: int) -> str:
                    return label_options[lb]
                answers = []
            elif 'UTCD' in dataset_name:
                descs = sconfig(f'UTCD.datasets.{dset_nm}.splits.{split}.labels')  # Descriptive labels
                n_cls = len(descs)
                # `label` is shared across all datasets, map to local label within dataset
                if self.cache_utcd is None:
                    path = os.path.join(utcd_util.get_output_base(), PROJ_DIR, DSET_DIR, 'processed', dataset_name)
                    # cos `Sequential`; each split, the label is the same
                    self.cache_utcd = datasets.load_from_disk(path)[split].features['labels'].feature
                # The ordering indicates int<=>str label mapping, i.e., index is int label,
                # see `process_utcd_dataset`

                def lb_int2desc(lb: int) -> str:
                    """
                    Map from local dataset label ordinal, in range(n_cls) to the descriptor
                    """
                    return descs[lb]
                answers = [self.cache_utcd.int2str(lb) for lb in labels]
            else:
                n_cls, label2description = (self.cache[dset_nm, split][k] for k in ('n_classes', 'label2description'))

                def lb_int2desc(lb: int) -> str:
                    return label2description[lb]

                if mode == 'inference':  # getting the answer doesn't matter here, see `evaluate_trained`
                    answers = []
                else:
                    raise NotImplementedError('Tokenization for non-UTCD datasets is not implemented yet')
                    answers = label2description[labels]

            idx_lbs = np.arange(n_cls)
            np.random.shuffle(idx_lbs)
            strs_lb = ' , '.join(f'" {lb_int2desc(idx)} "' for idx in idx_lbs)
            question = self.templates[idxs_tpl[i]].format(strs_lb)
            n_answs = len(answers)
            if n_answs > 1:
                idx_answs = np.arange(n_answs)
                np.random.shuffle(idx_answs)
                answers = [answers[idx] for idx in idx_answs]

            ids_ques = self._call_paren(question, **kwargs)
            ids_text = self._call_paren(text, **kwargs)
            id_sep = self.enc_spec(self.ques_sep_token)
            ids_answ = [self._call_paren(a, **kwargs) for a in answers]
            ids_answ = sum(join_it(ids_answ, [id_sep]), start=[])
            ln_q, ln_t, ln_a = len(ids_ques), len(ids_text), len(ids_answ)

            if mode == 'inference':
                # If text sample is so long that we need to truncate, leave room for one label only
                ln_cont = (1+ln_q+1) + (1+ln_t+1) + 1  # for `pref_answ`
                max_label_id_length = self.cache[dset_nm, split]['max_label_id_length']
                # The maximum number of tokens that could fit for context/prompt
                room = self.model_max_length-1 - max_label_id_length  # Also needs to generate `EOS`
                if ln_cont > room:
                    # Crop the text portion so that the longest label can be generated
                    ln_t_ = room - ((1+ln_q+1) + (1+1) + 1)
                    assert ln_t_ > 0
                    warn(f'Sample without answer longer than model max sequence length and dataset {dset_nm} labels: '
                         f'{ln_cont} > {self.model_max_length} - Text portion cropped: {ln_t} > {ln_t_} for inference')
                    ids_text = ids_text[:ln_t_]
            elif mode == 'train':
                ln_ids = ln_q + ln_t + ln_a
                if ln_ids > self.max_len_single_sentence:
                    # Crop the text portion, keep question and label intact,
                    # i.e., ensure no classification label is cropped
                    ln_t_ = self.max_len_single_sentence - (ln_q + ln_a)
                    assert ln_t_ > 0
                    warn(f'Sample with answer longer than model max sequence length for dataset {dset_nm}: '
                         f'{ln_ids+6} > {self.model_max_length} - Text portion cropped: {ln_t} > {ln_t_} for training')
                    ids_text = ids_text[:ln_t_]
            # else, `stats`, no truncation
            # Number of contex tokens, up until answer token, inclusive
            n_ques, n_text, n_answ = (1+len(ids_ques)+1), (1+len(ids_text)+1), (1+len(ids_answ)+1)
            n_cont = n_ques + n_text + 1
            ids = [
                self.enc_spec(self.boq_token), *ids_ques, self.enc_spec(self.eos_token),
                self.enc_spec(self.bot_token), *ids_text, self.enc_spec(self.eos_token),
                self.enc_spec(self.boa_token), *ids_answ, self.enc_spec(self.eos_token)
            ]
            tids = [self.enc_spec(self.question_type_token)] * n_ques + \
                   [self.enc_spec(self.text_type_token)] * n_text + \
                   [self.enc_spec(self.answer_type_token)] * n_answ
            if mode == 'inference':
                ids, tids = ids[:-(n_answ-1)], tids[:-(n_answ-1)]
                assert len(ids) == (n_ques+n_text+1)  # sanity check
            msks = [1] * len(ids)  # Encode ids are attended for CLM
            # Context position ids, followed by output position ids
            # adding `n_token` offset for the modified positional embeddings, see `ZsGPT2Model`
            pids = list(range(n_cont)) + [i + n_token for i in range(len(ids)-n_cont)]
            assert all(len(lst_ids) == len(ids) for lst_ids in (ids, tids, msks, pids))  # Sanity check

            def pad(ints: List[int], name) -> List[int]:
                """
                Pad to max_length, truncate if necessary
                """
                if name == 'attention_mask':
                    int_pad = 0  # Ignore in attention
                elif name == 'position_ids':
                    # Arbitrary, since will be ignored, but needs to be within `n_token` for embedding mapping
                    int_pad = 0
                else:
                    # `input_id`s set to `pad_token` will be ignored by `DataCollatorForLanguageModeling`
                    int_pad = self.enc_spec(self.pad_token)
                return ints[:max_length] if len(ints) > max_length else (ints + [int_pad] * (max_length - len(ints)))
            out = {k: (pad(ints, k) if mode == 'train' else ints) for k, ints in ((
                ('input_ids', ids), ('attention_mask', msks), ('token_type_ids', tids), ('position_ids', pids)
            ))}
            out['dataset_id'] = dataset_id  # For computing zero-shot classification accuracy
            if mode == 'stats':  # the number of tokens for just the text part
                out['ids_text'] = ids_text
            return out
        # See `zeroshot_encoder.util.util.py::process_utcd_dataset`
        keys_ = ['dataset_id', 'text', 'labels', 'label_options']
        if mode == 'inference-sample':
            assert not is_batched, f'Batched {logi("inference-sample")} not supported'
        else:
            assert 'label_options' not in samples, \
                f'{logi("label_options")} supported for {logi("inference-sample")} only'
        if is_batched:
            ds = [call_single(i, d_id, txt, lb) for i, (d_id, txt, lb) in enumerate(zip(
                *[samples[k] for k in keys_ if k in samples]  # only `label_options` may not be required
            ))]
            return BatchEncoding({k: [d[k] for d in ds] for k in ds[0]})  # Stack all the ids
        else:
            return BatchEncoding(call_single(0, *[samples[k] for k in keys_]))


class ZsGPT2Model(GPT2Model):
    """
    Modifying the `GPT2Model` for 0-shot classification paper
    """
    def __init__(self, config_):
        super().__init__(config_)
        # Override internal state, instead of adding internal state, so that forward pass stays untouched
        # Double the positional embedding matrix, as if stacking the context & output embedding matrices together
        # See positional id assignment in `ZsGPT2Tokenizer`
        self.wpe = nn.Embedding(config_.max_position_embeddings*2, self.embed_dim)


def pprint_gpt2_input(tokenizer: ZsGPT2Tokenizer, d: Dict[str, torch.Tensor]):
    """
    Prints to console the encoded ids, positional ids and type ids as sanity check
    """
    n_ct, n_dnm, n_wd = 3, 10, 13
    n_pad = n_ct + n_dnm + 3
    ids, pids, tids, dids = (d[k].detach() for k in ('input_ids', 'position_ids', 'token_type_ids', 'dataset_id'))
    pad = tokenizer.enc_spec(tokenizer.pad_token)
    id2name = sconfig('UTCD.dataset_id2name')

    for i, (ids_, did, pids_, tids_) in enumerate(zip(ids, dids, pids, tids)):
        msk = (ids_ != pad)
        ids_, pids_, tids_ = ids_[msk], pids_[msk], tids_[msk]
        print(f'{i:>{n_ct}}: {id2name[did.item()]:>{n_dnm}}', end=' ')
        for id_ in ids_:
            tok = tokenizer.decode(id_)
            print(f'{tok:>{n_wd}}', end='')
        print()

        print(' ' * n_pad, end='')
        for pid in pids_:
            print(f'{pid.item():>{n_wd}}', end='')
        print()
        print(' ' * n_pad, end='')
        for tid in tids_:
            print(f'{tokenizer.decode(tid):>{n_wd}}', end='')
        print()


class ZsGPT2LMHeadModel(GPT2LMHeadModel):
    """
    So that `ZsGPT2Model` is loaded
    """
    def __init__(self, config_):
        super().__init__(config_)
        self.transformer = ZsGPT2Model(config_)  # Override internal state

    def forward(self, dataset_id=None, **kwargs):
        # Function override to ignore `dataset_id`, not need in learning; Just need to pass value for evaluation
        # if torch.any(kwargs['input_ids'] == self.tokenizer.encode(self.tokenizer.ques_sep_token)[0]):
        # pprint_gpt2_input(self.tokenizer, d=kwargs | dict(dataset_id=dataset_id))
        # exit(1)
        return super().forward(**kwargs)

    @classmethod
    def from_pretrained(cls, *args, is_zs_gpt2: bool = False, **kwargs):
        """
        :param is_zs_gpt2: If True, loads a local `ZsGPT2LMHeadModel`; otherwise, expects a GPT2 model
        """
        if is_zs_gpt2:
            return super().from_pretrained(*args, **kwargs)
        else:
            md_ = super().from_pretrained(*args, **kwargs)  # Loads the GPT2LMHeadModel while ignoring `wpe.weight`
            md_ori = GPT2LMHeadModel.from_pretrained(*args, **kwargs)
            weight_pretrained = md_ori.transformer.wpe.state_dict()['weight']
            # Check `vars(md_ori.transformer.wpe)`, weight is the only parameter
            del md_ori

            # Crude loading the pretrained weights, to each half of the doubled positional embedding
            with torch.no_grad():
                n_tok = md_.transformer.wpe.weight.shape[0]
                if n_tok == 1024 * 2:
                    md_.transformer.wpe.weight[:1024, :] = weight_pretrained
                    md_.transformer.wpe.weight[1024:, :] = weight_pretrained
                else:
                    warn('Wrong model size, positional not loaded. This is expected in debugging')
            return md_

    @staticmethod
    def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
        """
        The original implementation is fine,
        cos in the 1st generation forward call, the positional ids are range(n) anyway
        but modify anyway just to be sure
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        # ========================== Begin of modified ==========================
        # else:  # Basically, keep the position ids
        #     position_ids = None
        # ========================== End of modified ==========================

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            # ========================== Begin of added ==========================
            'dataset_id': kwargs['dataset_id']  # Should definitely exist
            # ========================== End of added ==========================
        }

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # ========================== Begin of added ==========================
        assert 'position_ids' in model_kwargs
        position_ids = model_kwargs['position_ids']
        is_1st_call = position_ids[0, 0] == 0  # 1st call to prepping inputs, should start the answer position_ids
        if is_1st_call:
            assert torch.all(position_ids[:, 0] == 0).item()  # Sanity check
        new_col = position_ids[:, -1]+1  # Increment the last position
        if is_1st_call:
            new_col = torch.zeros_like(new_col) + self.config.n_ctx  # Per the paper, generating answer now
        # model_kwargs['position_ids'] = torch.cat([position_ids, new_col.unsqueeze(-1)], dim=-1)
        # Integrate with `past_key_values`,
        # Inspired by `GPT2LMHeadModel.prepare_inputs_for_generation`, looks like keep only the new column
        model_kwargs['position_ids'] = new_col.unsqueeze(-1)
        # ========================== End of added ==========================
        return model_kwargs


def tokenize_func(
        tokenizer: ZsGPT2Tokenizer, dataset_name='ag_news', max_length=None,
        split: str = 'train', mode: str = 'train', **kwargs
):
    def _tokenize_func(sample: Dict[str, List]):
        """
        :param sample: A batch of data samples
        """
        if 'UTCD' not in dataset_name:
            sample['dataset_id'] = [sconfig('UTCD.dataset_name2id')[dataset_name]] * len(sample['text'])
        # Otherwise, `dataset_id` already part of input
        return tokenizer(sample, dataset_name=dataset_name, max_length=max_length, split=split, mode=mode, **kwargs)
    return _tokenize_func


def get_model_n_tokenizer(model_name='gpt2', save_gpu_memory: bool = True) -> Tuple[
    ZsGPT2LMHeadModel, ZsGPT2Tokenizer, DataCollatorForLanguageModeling
]:
    pretrained_model_name = 'gpt2'

    if 'debug' in model_name:  # Try a smaller model for training sanity check
        if 'large' in model_name:
            n_token = 128
        else:
            n_token = 4
        conf = AutoConfig.from_pretrained('gpt2')
        # If using cpu, must be debugging and hence no `gradient_checkpointing`, see `get_train_setup`
        conf.update(dict(n_ctx=n_token, n_positions=n_token, use_cache=not torch.cuda.is_available()))
        model_ = ZsGPT2LMHeadModel.from_pretrained(pretrained_model_name, config=conf, ignore_mismatched_sizes=True)
        model_max_length = n_token
    else:
        model_max_length = 1024  # Keep max seq len of 1024, instead of 512 in paper, for longer texts & more labels
        conf = AutoConfig.from_pretrained(model_name)
        # `use_cache` in compatible with `gradient_checkpointing`, see `get_train_setup`
        conf.update(dict(use_cache=not (torch.cuda.is_available() and save_gpu_memory)))
        # Keep the 1024 token length, reducing to 512 tokens involves loading part of pretrained weights, complicated
        model_ = ZsGPT2LMHeadModel.from_pretrained(model_name, config=conf, ignore_mismatched_sizes=True)

    tokenizer_ = ZsGPT2Tokenizer.from_pretrained(
        pretrained_model_name, use_fast=True, model_max_length=model_max_length
    )
    model_.resize_token_embeddings(len(tokenizer_))
    model_.tokenizer = tokenizer_

    return model_, tokenizer_, DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False)


def get_train_setup(
        model_name='gpt2', do_eval=True, train_args: Dict = None,
        save_gpu_memory: bool = True
) -> TrainingArguments:
    name_ = model_name
    if name_ == 'debug-gpt-ori':
        name_ = 'gpt2'

    d_train_args = {
        'debug': dict(
            learning_rate=1e-4,
            batch_size=4,
            weight_decay=1e-2,
            num_train_epochs=4,
            lr_scheduler_type=SchedulerType.CONSTANT,
        ),
        'debug-large': dict(
            learning_rate=5e-5,
            batch_size=4,
            weight_decay=1e-2,
            num_train_epochs=40,
            lr_scheduler_type=SchedulerType.CONSTANT,
        ),
        'gpt2': dict(
            learning_rate=3e-5,
            batch_size=32,
            weight_decay=1e-2,
            num_train_epochs=5,
            lr_scheduler_type=SchedulerType.COSINE,
        ),
        'gpt2-medium': dict(
            learning_rate=4e-5,
            train_batch_size=128,
            eval_batch_size=64,
            gradient_accumulation_steps=1,
            weight_decay=1e-2,
            num_train_epochs=10,
            lr_scheduler_type=SchedulerType.COSINE,
        )
    }
    lr, bsz, decay, n_ep, sch, gas = (d_train_args[name_].get(k, None) for k in [
        'learning_rate', 'batch_size', 'weight_decay',
        'num_train_epochs', 'lr_scheduler_type', 'gradient_accumulation_steps'
    ])
    if bsz is None:
        bsz_tr, bsz_vl = (d_train_args[name_].get(k, None) for k in ('train_batch_size', 'eval_batch_size'))
        assert bsz_tr is not None and bsz_vl is not None
    else:
        bsz_tr = bsz_vl = bsz
    args = dict(
        output_dir=os.path.join(utcd_util.get_output_base(), PROJ_DIR, MODEL_DIR, 'gpt2', model_name, now(for_path=True)),
        do_train=True,
        do_eval=do_eval,
        evaluation_strategy='steps' if do_eval else 'no',
        per_device_train_batch_size=bsz_tr,
        per_device_eval_batch_size=bsz_vl,
        gradient_accumulation_steps=gas,
        eval_accumulation_steps=128,  # Saves GPU memory
        # Adam's beta1, beta2, epsilon taken from the GPT2 config in
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
        learning_rate=lr,
        weight_decay=decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1,
        num_train_epochs=n_ep,
        lr_scheduler_type=sch,
        warmup_ratio=1e-2,
        log_level='info',
        # log_on_each_node=False,
        log_level_replica='info',
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        fp16=torch.cuda.is_available(),
        fp16_full_eval=False,
        # fp16_full_eval=False,  # As in doc, harms metric
        optim=OptimizerNames.ADAMW_TORCH,
        disable_tqdm=True,
        # Pass dataset name information down to `compute_loss` for computing text classification accuracy
        remove_unused_columns=False,
        report_to='none',
        # Set to True on CPU gives warning; Enable for fitting in `clarity1` memory
        gradient_checkpointing=torch.cuda.is_available() and save_gpu_memory
    )
    if train_args is None:
        train_args = dict()
    args = {k: v for k, v in args.items() if v is not None}
    args.update(train_args)
    return TrainingArguments(**args)


def compute_metrics(eval_pred: MyEvalPrediction):
    """
    Will be called on eval data only, **during training**
    """
    # Intended to work with `CustomTrainer.prediction_step`
    if not hasattr(compute_metrics, 'metric'):
        compute_metrics.metric = datasets.load_metric('accuracy')
    # Labels are per-sample already, see `CustomTrainer.prediction_step`
    preds, trues, dids = eval_pred.predictions, eval_pred.label_ids, eval_pred.dataset_ids
    return compute_metrics.metric.compute(predictions=preds, references=trues)


def get_all_setup(
        model_name, dataset_name: str = 'ag_news',
        n_sample=None, random_seed=None, do_eval=True, custom_logging=True,
        train_args: Dict = None, dataset_args: Dict = None,
        is_ddp: Union[bool, int] = False  # so that my own logging is correct
) -> Tuple[GPT2LMHeadModel, Union[GPT2TokenizerFast, ZsGPT2Tokenizer], datasets.Dataset, datasets.Dataset, Trainer]:
    if model_name == 'debug-gpt-ori':  # Sanity check: As if keep training GPT-2, with padding for simplicity
        conf = AutoConfig.from_pretrained('gpt2')
        conf.update(dict(use_cache=False))
        model_ = GPT2LMHeadModel.from_pretrained('gpt2', config=conf)
        tokenizer_ = GPT2TokenizerFast.from_pretrained('gpt2')
        data_collator_ = None
        train_args_ = get_train_setup(model_name, do_eval=do_eval)

        def group_texts(examples):
            examples = tokenizer_(examples['text'])
            # Taken from
            # https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb
            # block_size = tokenizer_.model_max_length
            block_size = 512  # To fit in memory
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result['labels'] = result['input_ids'].copy()
            return result
        tr_map_func = vl_map_func = group_texts
    else:
        save_gpu_mem = 'arc-ts' not in get_hostname()
        # save_gpu_mem = True  # Gradient checkpointing still needed - otherwise doesn't fit in 44G GPU
        model_, tokenizer_, data_collator_ = get_model_n_tokenizer(model_name, save_gpu_memory=save_gpu_mem)
        train_args_ = get_train_setup(model_name, do_eval=do_eval, train_args=train_args, save_gpu_memory=save_gpu_mem)
        tr_map_func = tokenize_func(tokenizer_, dataset_name=dataset_name, split='train')
        vl_map_func = tokenize_func(tokenizer_, dataset_name=dataset_name, split='test')

    if dataset_args is None:
        dataset_args = dict()
    dset_tr_, dset_vl_ = get_dataset(
        dataset_name=dataset_name,
        map_func=dict(train=tr_map_func, test=vl_map_func), remove_columns=['text', 'labels'],
        n_sample=n_sample, random_seed=random_seed,
        fast='debug' not in model_name, **dataset_args
    )
    trainer_args = dict(
        model=model_, args=train_args_, data_collator=data_collator_,
        train_dataset=dset_tr_, eval_dataset=dset_vl_, compute_metrics=compute_metrics
    )
    trainer_ = CustomTrainer(
        tokenizer=tokenizer_, custom_logging=custom_logging, compute_cls_acc=model_name != 'debug-gpt-ori',
        is_ddp=is_ddp,
        **trainer_args
    )
    return model_, tokenizer_, dset_tr_, dset_vl_, trainer_


def plot_dataset_token_length_stats(domain: str = 'in'):
    ca(dataset_domain=domain)
    tokenizer = get_model_n_tokenizer('gpt2-medium')[1]
    # `split` shouldn't matter
    func = tokenize_func(tokenizer=tokenizer, dataset_name=f'UTCD-{domain}', split='train', mode='stats')
    did2nm = sconfig('UTCD.dataset_id2name')

    def map_func(examples):
        tokenized = func(examples)
        return dict(
            n_token=[len(ids) for ids in tokenized['input_ids']],
            n_token_text=[len(ids) for ids in tokenized['ids_text']],
            dataset_name=[did2nm[i] for i in tokenized['dataset_id']]
        )
    dset_tr, dset_vl = get_dataset(
        dataset_name=f'UTCD-{domain}',
        map_func=map_func, remove_columns=['text', 'labels'],
        # n_sample=1024*16,
        # random_seed=77,
        fast=True
    )
    # discard training set for out-of-domain
    dset = datasets.concatenate_datasets([dset_tr, dset_vl]) if domain == 'in' else dset_tr
    df = pd.DataFrame(dset[:])
    ic(df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    args_bar = dict(kde=True, kde_kws=dict(bw_adjust=0.5, gridsize=2048))
    args_cum = dict(cumulative=True, fill=False, element='step')
    for i_row, i_col in itertools.product(range(2), range(2)):
        ax = axes[i_row, i_col]
        legend = i_row == 1 and i_col == 0
        args = dict(palette='husl', legend=legend, common_norm=False, ax=ax, stat='density')
        args |= args_bar if i_col == 0 else args_cum
        x = 'n_token' if i_row == 0 else 'n_token_text'
        if i_col == 0:
            n_bin = df[x].max() - df[x].min() + 1
            args['bins'] = n_bin
        sns.histplot(data=df, x=x, hue='dataset_name', **args)
        ax.set(xlabel='#token' if i_row == 0 else '#token for text', ylabel=None)
        p = norm().cdf(3.5)  # # dynamic upperbound; quantile by std
        mi, ma = df[x].min(), math.ceil(df[x].quantile(p))
        ax.set_xlim([mi, ma])
    title = f'GPT2 token length distribution for UTCD {domain}-domain'
    plt.suptitle(title)
    fig.supylabel('density')

    output_dir = os.path.join(BASE_PATH, PROJ_DIR, 'plot')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{title}, {now(for_path=True)}.png'), dpi=300)


def load_trained(epoch: int = 3) -> ZsGPT2LMHeadModel:
    assert epoch in [2, 3]
    if not hasattr(load_trained, 'epoch2path'):
        load_trained.epoch2path = {
            # 2: os.path.join('2022-03-04 21-33-12', 'checkpoint-37066'),
            # 3: os.path.join('2022-03-04 21-33-12', 'checkpoint-55599'),
            3: os.path.join('2022-04-02_11-51-19', 'checkpoint-51390'),
        }
    path = os.path.join(BASE_PATH, PROJ_DIR, 'trained-models', 'gpt2-nvidia', load_trained.epoch2path[epoch])
    return ZsGPT2LMHeadModel.from_pretrained(path, is_zs_gpt2=True).to('cuda')  # with caching


def evaluate_trained(domain: str = 'in', batch_size: int = 48, n_ep: int = 3):
    """
    Run evaluation, on potentially multi-label datasets
    """
    ca(domain=domain)
    model = load_trained(epoch=n_ep).to('cuda')
    conf, model_cnm = model.config, model.__class__.__qualname__
    # To disable warning `Setting `pad_token_id` to `eos_token_id` for open-end generation.`
    model_size = conf.max_length = conf.n_ctx
    conf.pad_token_id = conf.eos_token_id
    model.eval()
    tkzer = ZsGPT2Tokenizer.from_pretrained('gpt2', use_fast=True, model_max_length=model_size)
    model.tokenizer = tkzer  # See ZsGPT2LMHeadModel.forward() sanity check`

    split = 'test'
    path_dir = os.path.join(BASE_PATH, PROJ_DIR, 'evaluations', MODEL_NAME, now(for_path=True))

    dataset_names = [
        dnm for dnm in sconfig('UTCD.datasets').keys()
        if (sconfig(f'UTCD.datasets.{dnm}.domain') == domain)
    ]
    d_model = OrderedDict([('model name', model_cnm), ('trained #epoch', n_ep), ('model size', model_size)])
    d_eval = OrderedDict([
        ('max batch size', batch_size),
        ('datasets', dataset_names)
    ])
    domain = 'in-domain' if domain == 'in' else 'out-of-domain'
    logger_name = 'GPT2-NVIDIA Evaluation'
    logger = get_logger(logger_name, typ='stdout')
    logger_fl = get_logger(
        f'{logger_name} file-write', typ='file-write',
        file_path=os.path.join(path_dir, f'{logger_name}, bsz={batch_size}, {domain}.log')
    )
    logger.info(f'Running evaluation {logi(domain)} on model {log_dict(d_model)}, with {log_dict(d_eval)}... ')
    logger_fl.info(f'Running evaluation {domain} on model {log_dict_nc(d_model)}, with {log_dict_nc(d_eval)}... ')

    for dnm_ in dataset_names:
        d_info = sconfig(f'UTCD.datasets.{dnm_}.splits.{split}')
        lb2id = defaultdict(lambda: -1)  # If generated invalid descriptive label, will return -1
        labels = d_info['labels']
        # predictions and label descriptions all to lower case to be more lenient
        lb2id.update({lb.lower(): i for i, lb in enumerate(labels)})
        dset = get_dataset(  # Get evaluation set only
            dataset_name=dnm_, splits='test',
            map_func=dict(test=tokenize_func(tkzer, dataset_name=dnm_, split='test', mode='inference')),
            remove_columns='text', n_sample=None, from_disk=True  # keeps the `labels`
        )[0]

        # Batched generation that **doesn't take up padding** is not supported by HuggingFace
        n_dset = len(dset)
        trues, preds = np.empty(n_dset, dtype=int), np.empty(n_dset, dtype=int)
        len_ids = np.array([len(ids) for ids in dset[:]['input_ids']])
        uniq_lens = np.unique(len_ids)
        # Batches of likely different batch sizes
        ln2idxs = [np.where(len_ids == ln)[0] for ln in uniq_lens]
        idxs_batches = sum(  # Get batches of same length, with max batch size of `batch_size`
            (np.split(idxs, range(batch_size, idxs.size, batch_size)) if idxs.size > batch_size else [idxs]
             for idxs in ln2idxs),
            start=[]
        )
        n_bch = len(idxs_batches)
        logger.info(f'Running evaluation on dataset {logi(dnm_)}, with labels {log_dict(lb2id)}, '
                    f'of {logi(len(dset))} unique texts in {logi(n_bch)} batches... ')
        logger_fl.info(f'Running evaluation on dataset {dnm_}, with labels {log_dict_nc(lb2id)}, '
                       f'of {len(dset)} unique texts in {n_bch} batches... ')

        n_computed = 0
        for step, idxs in enumerate(idxs_batches):  # Each batch has input samples of the same token length
            idxs = [int(idx) for idx in idxs]  # `Dataset.select` works with `int` indices only
            inputs = {  # No need to pad; Don't need to the labels to complicate forward pass
                k: torch.tensor(v, device='cuda') for k, v in dset[idxs].items()
                if k != 'labels'  # Convert `dataset_id` too so that fits into HuggingFace APIs
            }
            outputs = model.generate(**inputs)  # Greedy decoding
            outputs_str = tkzer.batch_decode(outputs, skip_special_tokens=False)
            n_computed += len(idxs)

            def set_pred_n_true(generated: str, i_sample: int) -> Tuple[int, int]:
                idxs_boa = get_substr_indices(generated, s_sub=tkzer.boa_token)
                # there will be at least one index, as in prompt
                assert len(idxs_boa) >= 1
                # **try to be as lenient**: try to extract the text part if possible
                answer_with_eos = generated[idxs_boa[-1] + len(tkzer.boa_token):]
                if len(idxs_boa) > 1:
                    logger.warning(f'{logi(model_cnm)} generated {logi(len(idxs_boa))} boa_token '
                                   f'instead of {logi(1)} with [{logi(answer_with_eos)}]')
                    logger_fl.warning(f'{model_cnm} generated {len(idxs_boa)} boa_token '
                                      f'instead of {1} with [{answer_with_eos}]')
                assert len(idxs_boa) == 1
                idxs_eos = get_substr_indices(answer_with_eos, s_sub=tkzer.eos_token)
                # GPT2 would generate multiple `eos_token` for the samples in the batch that terminates early
                if len(idxs_eos) == 0:  # Still, **try to be as lenient**
                    logger.warning(f'{logi(model_cnm)} didn\'t finish generating answer '
                                   f'with [{logi(answer_with_eos)}]')
                    logger_fl.warning(f'{model_cnm} didn\'t finish generating answer with [{answer_with_eos}]')
                    answer = answer_with_eos
                else:
                    answer = answer_with_eos[:idxs_eos[0]]  # until the 1st eos
                answer = answer.lower()
                idxs_sep = get_substr_indices(answer, s_sub=tkzer.ques_sep_token)
                if len(idxs_sep) > 0:
                    answers = [answer[:idxs_sep[0]]]
                    for i, idx in enumerate(idxs_sep[:-1]):
                        answers.append(answer[idx + len(tkzer.ques_sep_token):idxs_sep[i+1]])
                    answers.append(answer[idxs_sep[-1] + len(tkzer.ques_sep_token):])
                    ic(answer, answers)
                    # exit(1)
                else:
                    answers = [answer]
                ids_pred = [lb2id[a] for a in answers]
                ids_true: List[int] = dset[i_sample]['labels']
                matched = set(ids_pred) & set(ids_true)
                if len(idxs_sep) > 0:
                    ic(ids_pred, ids_true, matched)
                    # exit(1)
                if len(matched) > 0:
                    # predicted label is one of the correct labels, pick that label so that prediction is correct
                    id_true = id_pred = next(iter(matched))
                else:
                    # prediction incorrect, pick a single label arbitrarily
                    # This renders class-level performance inaccurate; TODO?
                    id_pred, id_true = -1, ids_true[0]
                preds[i_sample], trues[i_sample] = id_pred, id_true
                return id_pred, id_true
            preds_batch, trues_batch = zip(*[
                set_pred_n_true(out, i_sample) for out, i_sample in zip(outputs_str, idxs)
            ])
            d_log = dict(
                step=f'{step+1:>{len(str(n_bch))}}/{n_bch}', progress=f'{n_computed:>{len(str(n_dset))}}/{n_dset}',
                sequence_length=len(inputs['input_ids'][0]),
                batch_size=f'{len(idxs):>{len(str(batch_size))}}/{batch_size}',
                n_acc=sum(p == t for p, t in zip(preds_batch, trues_batch)),
                ids_pred=list(preds_batch), ids_true=list(trues_batch)
            )
            logger.info(log_dict(d_log, with_color=True))
            logger_fl.info(log_dict(d_log, with_color=False))

        def check_labels_filled(lbs):  # sanity check, every index is assigned a label
            return np.all((-1 <= lbs) & (lbs < len(labels)))
        assert check_labels_filled(trues) and check_labels_filled(preds)
        df = pd.DataFrame(
            # note `-1` is not actual label, support of 0 - included for full label specification per sklearn
            # **note** cos the -1 label, the `macro avg` row is not accurate;
            # included it for getting global accuracy
            classification_report(
                trues, preds, labels=[-1, *range(len(labels))], target_names=['Label not in dataset', *labels],
                output_dict=True
            )
        ).transpose()
        path = os.path.join(path_dir, f'{dnm_}.csv')
        df.to_csv(path)
        logger.info(f'Evaluation on {logi(dnm_)} written to CSV at {logi(path)}')
        logger_fl.info(f'Evaluation on {dnm_} written to CSV at {path}')


def gpt2_inference(text: str, label_options: List[str]) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained(epoch=3).to(device)
    model_size = model.config.max_length = model.config.n_ctx
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    tkzer = ZsGPT2Tokenizer.from_pretrained('gpt2', use_fast=True, model_max_length=model_size)

    # 'dataset_name` just so that it passes, irrelevant
    tokenize_fn = tokenize_func(tkzer, dataset_name='UTCD', mode='inference-sample')
    inputs = tokenize_fn(dict(text=text, dataset_id=-1, labels=-1, label_options=label_options))
    inputs = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in inputs.items()}  # add dummy batch dim
    outputs = model.generate(**inputs)
    return tkzer.batch_decode(outputs, skip_special_tokens=False)


if __name__ == '__main__':
    from icecream import ic

    from zeroshot_encoder.util import *

    seed = sconfig('random-seed')
    transformers.set_seed(seed)

    def training():
        # dnm = 'ag_news'
        dnm = 'UTCD'

        # nm = 'debug'
        # nm = 'debug-gpt-ori'
        # nm = 'debug-large'
        # nm = 'gpt2'
        nm = 'gpt2-medium'

        # n = 1
        n = 128
        # n = 1024
        # n = 4500
        # n = 1024 * 32
        # n = None

        tr_args = None
        # tr_args = dict(num_train_epochs=32)

        md, tkzer, dset_tr, dset_vl, trainer = get_all_setup(
            nm, dnm, do_eval=False, custom_logging=True, n_sample=n, random_seed=seed, train_args=tr_args
        )

        def profile_train():
            profile_runtime(lambda: trainer.train())
        # profile_train()

        def train_(resume=False):
            if resume:
                checkpoint_path = '/scratch/profmars_root/profmars0/stefanhg/Zero-shot-text-classification/' \
                                  'models/gpt2/gpt2-medium/2022-03-03 00-23-41/checkpoint-18533'
                trainer.train(checkpoint_path)  # Resume from checkpoint
            else:
                trainer.train()
            trainer.save_model(os.path.join(trainer.args.output_dir))
            # trainer.evaluate()
        # train_(resume=False)
        # train(resume=True)
    # training()

    def evaluating():
        def profile_evaluation():
            profile_runtime(lambda: evaluate_trained(domain='in', batch_size=48), sleep=2)
        # profile_evaluation()

        evaluate_trained(domain='out', batch_size=48)
    # evaluating()

    def new_training():
        dnm = 'UTCD-in'
        nm = 'gpt2-medium'
        # n = 128
        # n = 32
        n = None
        # n = 256

        # train_args = dict(  # overfit small
        #     weight_decay=0,
        #     learning_rate=3e-4,
        #     per_device_train_batch_size=4,
        #     num_train_epochs=128,
        #     lr_scheduler_type=SchedulerType.CONSTANT,
        #     gradient_accumulation_steps=8,
        #     save_strategy='no',
        # )
        dataset_args = None
        # dataset_args = dict(  # debugging
        #     # filter_func=lambda sample: sample['dataset_id'] == config('UTCD.dataset_name2id')['dbpedia'],
        #     filter_func=lambda sample: len(sample['labels']) > 1,
        # )
        # train_args = dict(
        #     num_train_epochs=3,
        #     per_device_train_batch_size=4,
        # gradient_accumulation_steps = 8
        # )
        train_args = dict(  # Distribute among GPUs & fit in memory; Effectively batch size 128 as in paper
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8
        )
        md, tkzer, dset_tr, dset_vl, trainer = get_all_setup(
            nm, dnm, do_eval=False, custom_logging=True, n_sample=n, random_seed=seed,
            train_args=train_args, dataset_args=dataset_args,
            is_ddp=4
        )
        trainer.train()
    # new_training()

    def sanity_check_trained_generate():
        text = 'hello world'
        label_options = ['happy', 'sad', 'angry', 'fearful', 'surprised']
        ic(text, label_options)
        ic(gpt2_inference(text, label_options))
    # sanity_check_trained_generate()

    plot_dataset_token_length_stats(domain='in')
