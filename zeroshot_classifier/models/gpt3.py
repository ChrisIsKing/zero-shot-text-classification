import re
import os
import json
import time
import random
import logging
import requests
from os.path import join as os_join
from typing import List, Dict, Any, Union, Optional
from argparse import ArgumentParser
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import datasets
import openai
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_dataset


__all__ = ['ApiCaller', 'PromptMap', 'evaluate']


logger = get_logger('GPT3')


class ApiCaller:
    """
    Make API call to Open API GPT3 model to get completions

    """
    url = 'https://api.openai.com/v1/completions'

    with open(os_join(u.proj_path, 'auth', 'open-ai.json')) as f:
        auth = json.load(f)
        # api_key, org = auth['api-key'], auth['organization']
        api_key, org = auth['api-key-chris'], auth['organization']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'OpenAI-Organization': org
    }

    def __init__(self, model: str = 'text-ada-001'):
        self.model = model

    def __call__(self, prompt: str, rand_sleep: bool = True, **kwargs) -> str:
        payload = dict(
            model=self.model,
            temperature=0,  # Generate w/ greedy decoding
            stop='\n',
            max_tokens=32
        )
        payload['prompt'] = prompt
        payload.update(kwargs)

        def _call():
            return requests.post(self.url, headers=self.headers, json=payload)

        if rand_sleep:  # Intended for concurrent requests, see evaluate `concurrent` flag
            time.sleep(random.uniform(0, 4))

        res = None
        i = 0
        sleep_time = 4
        while not res or res.status_code != 200:
            if res:
                assert res.status_code == 429  # Too many request, retry
            if i % 4 == 0:  # Wait for `Too Many Requests` to pass
                sleep_time *= 2

            logger.info(f'Too Many Requests, retrying in {pl.i(sleep_time)}s... ')
            time.sleep(sleep_time)
            res = _call()
            i += 1

        res = json.loads(res.text)
        assert len(res['choices']) == 1  # sanity check only generated one completion
        return res['choices'][0]['text']


def text2n_token(txt: str) -> int:
    if not hasattr(text2n_token, 'token_pattern'):
        text2n_token.token_pattern = re.compile(r'(?u)\b\w+\b')  # taken from sklearn.CountVectorizer
    return len(text2n_token.token_pattern.findall(txt))


def truncate_text(text: str = None, n: int = None) -> str:
    return ' '.join(text.split()[:n])


class PromptMap:
    """
    Create the GPT3 prompt given text and label options

    Since we don't know number of tokens the prompt is tokenized into,
        set artificial length limit based on number of words
    """
    templates = sconfig('baselines.gpt2-nvidia.templates')
    n_template = len(templates)

    logger = get_logger('Prompt Map')

    def __init__(
            self, dataset_name: str = None, max_text_length: int = 1024, max_prompt_length: int = 1024 + 256,
            logger_fl: logging.Logger = None
    ):
        self.dataset_name = dataset_name
        self.labels = sconfig(f'UTCD.datasets.{dataset_name}.splits.test.labels')  # Take labels from the test split

        self.n_cls = len(self.labels)

        self.max_text_length = max_text_length
        self.max_prompt_length = max_prompt_length

        self.logger_fl = logger_fl
        d_log = {
            'dataset_name': dataset_name, 'labels': self.labels, '#class': self.n_cls,
            'max_text_length': max_text_length, 'max_prompt_length': max_prompt_length
        }
        PromptMap.logger.info(f'Prompt Map initialized with: {pl.i(d_log)}')
        if self.logger_fl:
            self.logger_fl.info(f'Prompt Map initialized with: {pl.nc(d_log)}')

    def __call__(self, text: str = None):
        n_txt = text2n_token(text)
        if n_txt > self.max_text_length:
            text = truncate_text(text, n_txt)
            PromptMap.logger.warning(f'Text too long and truncated: {pl.i(n_txt)} -> {pl.i(self.max_text_length)}')

        idx_lbs = np.arange(self.n_cls)
        np.random.shuffle(idx_lbs)  # The order which the labels appears are random
        label_options_str = ' , '.join(f'" {self.labels[idx]} "' for idx in idx_lbs)

        idx_tpl = np.random.randint(PromptMap.n_template)
        question = PromptMap.templates[idx_tpl].format(label_options_str)
        prompt = self._to_prompt(question=question, text=text)

        n_prompt = text2n_token(prompt)
        if n_prompt > self.max_prompt_length:
            n_txt_ = self.max_prompt_length - text2n_token(question) - 2  # 2 for `Text` and `Answer`
            assert n_txt_ >= 50  # sanity check
            text = truncate_text(text, n_txt_)
            PromptMap.logger.warning(f'Prompt too long and text segment truncated: '
                                     f'{pl.i(n_prompt)} -> {pl.i(self.max_prompt_length)}')

            if self.logger_fl:
                self.logger_fl.warning(f'Prompt too long and text segment truncated: '
                                       f'{pl.nc(n_prompt)} -> {pl.nc(self.max_prompt_length)}')
            prompt = self._to_prompt(question=question, text=text)
        return prompt

    @staticmethod
    def _to_prompt(question: str = None, text: str = None):
        # return f'Text: {text}\nQuestion: {question}\n Answer:'
        return f'{question}\n Text: {truncate_text(text)} \n Answer:'


@dataclass
class GPT3EvalMeta:
    text: str = None
    prompt: str = None
    generated: str = None


@dataclass
class _EvalSingleOut:
    pred: int = None
    true: int = None
    meta: GPT3EvalMeta = None


class _EvalSingle:
    def __init__(
            self, pm: PromptMap = None, api_caller: ApiCaller = None,
            label_options: List[str] = None, lb2id: Dict[str, int] = None,
            logger_fl: logging.Logger = None, return_text: bool = False
    ):
        self.pm = pm
        self.ac = api_caller
        self.label_options = label_options
        self.lb2id = lb2id

        self.logger_fl = logger_fl

        self.return_text = return_text

    def __call__(self, e: Dict[str, Any], pbar=None) -> _EvalSingleOut:
        txt, lbs = e['text'], e['labels']
        prompt = self.pm(txt)
        answer = self.ac(prompt)  # TODO: maybe GPT3 generates multiple answers?
        answer = answer.lower().strip()

        if pbar:
            _d_log = dict(labels=[self.label_options[i] for i in lbs], answer=[answer])
            pbar.set_postfix({k: pl.i(v) for k, v in _d_log.items()})

        ret: Dict[str, Any]
        if answer in self.label_options:
            ret = dict(pred=self.lb2id[answer], true=self.lb2id[answer])
        else:
            logger.warning(f'Generated {pl.i([answer])}, not one of label options')
            self.logger_fl.warning(f'Generated {pl.nc([answer])}, not one of label options')

            ret = dict(pred=-1, true=lbs[0])
        if self.return_text:
            ret.update(meta=GPT3EvalMeta(text=txt, prompt=prompt, generated=answer))
        return _EvalSingleOut(**ret)


def evaluate(
        model: str = 'text-ada-001', domain: str = 'in', dataset_name: str = 'all', concurrent: bool = False,
        subsample: Union[bool, int] = False, subsample_seed: int = 77, store_meta: bool = False,
        store_frequency: Optional[int] = None, resume: List[str] = None
):
    ac = ApiCaller(model=model)

    if dataset_name == 'all' and subsample:
        raise NotImplementedError('Subsampling intended for single dataset')
    dataset_names = utcd_util.get_eval_dataset_names(domain=domain, dataset_name=dataset_name)

    _preds, _trues, _infs = None, None, None
    if resume:
        assert len(dataset_names) == 1  # sanity check, intended for resuming from single dataset
        _preds, _trues, _infs = [], [], []
        for r in resume:
            with open(r, 'r') as fl:
                meta = json.load(fl)
            _preds.extend(meta['preds'])
            _trues.extend(meta['trues'])
            _infs.extend(meta['inferences'])
        assert len(_preds) == len(_trues) == len(_infs)  # sanity check

    d = dict(md=model, dm=domain, dnm=dataset_name)
    output_dir_nm = f'{now(for_path=True)}_Zeroshot-GPT3-Eval_{pl.pa(d)}'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))

    logger_fl = get_logger('GPT3 Eval', kind='file-write', file_path=os_join(output_path, f'eval.log'))
    d_log: Dict[str, Any] = dict(model_name=model, domain=domain, dataset_names=dataset_names, output_path=output_path)
    d_log.update(dict(concurrent=concurrent, subsample=subsample, subsample_seed=subsample_seed, store_meta=store_meta))
    d_log['store_frequency'] = store_frequency
    logger.info(f'Evaluating GPT3 model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating GPT3 model w/ {d_log}... ')

    os.makedirs(output_path, exist_ok=True)

    for dnm in dataset_names:
        if subsample:
            n_tgt = 5000 if isinstance(subsample, bool) else subsample
            dset = utcd_util.subsample_dataset(dataset_name=dnm, split='test', n_tgt=n_tgt, seed=subsample_seed)
        else:
            dset = get_dataset(dnm, splits='test')['test']
        dset: datasets.Dataset

        if resume:
            n_ori = len(dset)
            ran_txts = set(e['text'] for e in _infs)

            def filt(sample: Dict[str, Any]) -> bool:
                return sample['text'] not in ran_txts
            dset = dset.filter(filt)
            # sanity check, every text completed should be accounted for, exactly once
            mic(len(dset), len(_infs), n_ori)
            assert len(dset) + len(_infs) == n_ori
            logger.info(f'{pl.i(len(_infs))} texts evaluated, resuming from {pl.i(n_ori)} => {pl.i(len(dset))} ')
            logger_fl.info(f'{len(_infs)} texts evaluated, resuming from {n_ori} => {len(dset)} ')

        pm = PromptMap(dataset_name=dnm, logger_fl=logger_fl)
        label_options = [lb.lower() for lb in pm.labels]
        lb2id = {lb: idx for idx, lb in enumerate(label_options)}
        args = dict(pm=pm, api_caller=ac, label_options=label_options, lb2id=lb2id, logger_fl=logger_fl)
        eval_single = _EvalSingle(**args, return_text=store_meta)

        lst_meta = []
        meta_path = os_join(output_path, f'{dnm}_meta.json')

        store_frequency = store_frequency or 100

        def write_meta():  # Writing completed inferences to file periodically, in case GPT3 eval gets stuck
            if len(lst_meta) % store_frequency == 0:
                logger.info(f'Writing eval instances to {pl.i(meta_path)}...')
                logger_fl.info(f'Writing eval instances to {meta_path}...')
                d_ = d_log
                infs = [asdict(m) for m in lst_meta]
                if resume:
                    infs = _infs + infs
                n = len(infs)
                if not concurrent:  # a numpy array created
                    __preds = list(preds[:n])
                    __trues = list(trues[:n])
                else:
                    __preds, __trues = preds, trues
                assert len(__preds) == len(__trues) == len(infs)  # sanity check
                d_.update(dict(inferences=infs, n=n, preds=__preds, trues=__trues))

                with open(meta_path, 'w') as f_:
                    json.dump(d_, f_, indent=4)

        if concurrent:
            with_tqdm = dict(desc=f'Evaluating {pl.i(dnm)}', total=len(dset))
            preds, trues = (_preds, _trues) if resume else ([], [])
            # order irrelevant
            for e in conc_yield(eval_single, dset, with_tqdm=with_tqdm, mode='thread', n_worker=4):
                preds.append(e.pred)
                trues.append(e.true)
                if store_meta:
                    lst_meta.append(e.meta)
                    write_meta()
        else:
            n_dset = len(dset)
            trues, preds = np.empty(n_dset, dtype=int), np.empty(n_dset, dtype=int)
            if resume:
                trues[:len(_trues)] = _trues
                preds[:len(_preds)] = _preds
            it = tqdm(dset, desc=f'Evaluating {pl.i(dnm)}')
            for idx, elm in enumerate(it):
                out = eval_single(elm, pbar=it)
                preds[idx], trues[idx] = out.pred, out.true
                if store_meta:
                    lst_meta.append(out.meta)
                    write_meta()

        args = dict(
            labels=[-1, *range(len(label_options))], target_names=['Label not in dataset', *label_options],
            zero_division=0, output_dict=True
        )
        report = classification_report(trues, preds, **args)
        acc = f'{report["accuracy"]:.3f}'
        logger.info(f'{pl.i(dnm)} accuracy: {pl.i(acc)}')
        logger_fl.info(f'{dnm} accuracy: {acc}')

        csv_path = os_join(output_path, f'{dnm}.csv')
        pd.DataFrame(report).transpose().to_csv(csv_path)


if __name__ == '__main__':
    mic.output_width = 256

    with open(os_join(u.proj_path, 'auth', 'open-ai.json')) as f:
        auth = json.load(f)
        org = auth['organization']
        api_key = auth['api-key']
        # api_key = auth['api-key-chris']
    openai.api_key = api_key

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'OpenAI-Organization': org
    }

    def check_api():
        res = requests.get(url='https://api.openai.com/v1/models', headers=headers)
        mic(res)
        res = json.loads(res.text)
        mic(res)
    # check_api()

    def try_completion():
        # model = 'text-ada-001'  # fastest
        model = 'text-davinci-002'  # most powerful

        payload = dict(
            model=model,
            prompt=["Say this is a test", "Say Happy"],
            max_tokens=6,
            temperature=0  # Generate w/ greedy decoding
        )
        res = requests.post(url='https://api.openai.com/v1/completions', headers=headers, json=payload)
        mic(res)
        res = json.loads(res.text)
        mic(res)
    # try_completion()

    def try_open_ai_api():
        # ppt = "def magic_function():\n\t"
        ppt = ["def magic_function():\n\t", "print('Hello world!')\n\t"]

        res = openai.Completion.create(
            model="curie",
            prompt=ppt,
            max_tokens=10,
        )
        mic(res)
    # try_open_ai_api()

    def try_open_ai_delay():
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(**kwargs):
            return openai.Completion.create(**kwargs)

        res = completion_with_backoff(
            model="text-davinci-002", prompt=["Once upon a time,", "there was a dog"], max_tokens=10
        )
        mic(res)
    # try_open_ai_delay()

    # evaluate(model='text-ada-001', domain='in', dataset_name='emotion')
    # evaluate(model='text-curie-001', domain='in', dataset_name='emotion', concurrent=True)
    # evaluate(model='text-davinci-002', domain='in', dataset_name='emotion')
    # evaluate(model='text-curie-001', domain='out', dataset_name='multi_eurlex', concurrent=True)

    # evaluate(model='text-curie-001', domain='in', dataset_name='finance_sentiment', concurrent=True)
    # evaluate(model='text-curie-001', domain='in', dataset_name='banking77', concurrent=True, subsample=True)
    # evaluate(model='text-davinci-002', domain='out', dataset_name='finance_sentiment')
    # evaluate(model='text-davinci-002', domain='out', dataset_name='consumer_finance')
    # evaluate(model='text-davinci-002', domain='out', dataset_name='amazon_polarity', concurrent=True)

    run_args = dict(model='text-curie-001', concurrent=True, subsample=True, store_meta=True, store_frequency=10)
    # dnm = 'amazon_polarity'
    # dnm = 'yelp'
    dnm_ = 'consumer_finance'
    rsm = [os_join(
        u.eval_path, '2022-11-17_23-41-12_Zeroshot-GPT3-Eval_{md=text-curie-001, dm=out, dnm=consumer_finance}',
        '22-11-17_out-of-domain', 'consumer_finance_meta.json'
    )]
    # evaluate(
    #     domain='out', dataset_name=dnm_, **run_args,
    #     resume=rsm
    # )

    def parse_args():
        parser = ArgumentParser()

        models = ['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002']
        parser.add_argument('--model', type=str, choices=models, default='text-ada-001', help="""
            GPT3 model from Open AI API, see `https://beta.openai.com/docs/models/gpt-3`
        """)
        parser.add_argument('--dataset', type=str, default='all', help="""
            One of dataset name in UTCD or `all` for all datasets in a domain, see argument `domain`
        """)
        parser.add_argument('--domain', type=str, choices=['in', 'out'], default='in', help="""
            One of [`in`, `out`] for in-domain, out-of-domain respectively
        """)
        parser.add_argument('--concurrent', type=bool, default=False, help="""
            Make GPT3 completion requests concurrently
        """)
        parser.add_argument('--subsample', type=int, default=5000, help="""
            Total #sample to subsample from the dataset
        """)
        parser.add_argument('--subsample_seed', type=int, default=77)
        return parser.parse_args()

    def command_prompt():
        args = parse_args()
        evaluate(
            model=args.model, dataset_name=args.dataset, domain=args.domain, concurrent=args.concurrent,
            subsample=args.subsample, subsample_seed=args.subsample_seed
        )
    # command_prompt()
