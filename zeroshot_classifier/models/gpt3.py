import logging
import re
import os
import json
import time
from typing import List, Dict, Any
from argparse import ArgumentParser

import requests
from os.path import join as os_join

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_dataset


logger = get_logger('GPT3')


class ApiCaller:
    """
    Make API call to Open API GPT3 model to get completions

    """
    url = 'https://api.openai.com/v1/completions'

    with open(os_join(u.proj_path, 'auth', 'open-ai.json')) as f:
        d = json.load(f)
        api_key, org = d['api-key'], d['organization']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'OpenAI-Organization': org
    }

    def __init__(self, model: str = 'text-ada-001'):
        self.model = model

    def __call__(self, prompt: str, **kwargs) -> str:
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
        res = None
        while not res or res.status_code != 200:
            if res:
                assert res.status_code == 429  # Too many request, retry
            time.sleep(0.5)
            res = _call()

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
            PromptMap.logger.warning(f'Text too long and truncated: {n_txt} -> {self.max_text_length}')

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


class _EvalSingle:
    def __init__(
            self, pm: PromptMap = None, api_caller: ApiCaller = None,
            label_options: List[str] = None, lb2id: Dict[str, int] = None,
            logger_fl: logging.Logger = None
    ):
        self.pm = pm
        self.ac = api_caller
        self.label_options = label_options
        self.lb2id = lb2id

        self.logger_fl = logger_fl

    def __call__(self, e: Dict[str, Any], pbar=None):
        txt, lbs = e['text'], e['labels']
        prompt = self.pm(txt)
        answer = self.ac(prompt)  # TODO: maybe GPT3 generates multiple answers?
        answer = answer.lower().strip()

        if pbar:
            _d_log = dict(labels=[self.label_options[i] for i in lbs], answer=[answer])
            pbar.set_postfix({k: pl.i(v) for k, v in _d_log.items()})

        if answer in self.label_options:
            return self.lb2id[answer], self.lb2id[answer]
        else:
            logger.warning(f'Generated {pl.i([answer])}, not one of label options')
            self.logger_fl.warning(f'Generated {pl.nc([answer])}, not one of label options')
            return -1, lbs[0]


def evaluate(model: str = 'text-ada-001', domain: str = 'in', dataset_name: str = 'all', concurrent: bool = False):
    ac = ApiCaller(model=model)

    all_dset = dataset_name == 'all'
    if not all_dset:
        _dom = sconfig(f'UTCD.datasets.{dataset_name}.domain')
        if domain is not None:
            domain = _dom
        else:
            assert domain == _dom
    if all_dset:
        dataset_names = utcd_util.get_dataset_names(domain)
    else:
        dataset_names = [dataset_name]

    output_dir_nm = f'{now(for_path=True)}_Zeroshot-GPT3-{model}'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))
    os.makedirs(output_path, exist_ok=True)

    log_fnm = f'{now(for_path=True)}_GPT3_{model}_{domain}_{dataset_name}_Eval'
    logger_fl = get_logger('GPT3 Eval', typ='file-write', file_path=os_join(output_path, f'{log_fnm}.log'))

    d_log = dict(model_name=model, domain=domain, dataset_names=dataset_names, output_path=output_path)
    logger.info(f'Evaluating GPT3 model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating GPT3 model w/ {d_log}... ')

    for dnm in dataset_names:
        dset = get_dataset(dnm, splits='test')['test']
        pm = PromptMap(dataset_name=dnm, logger_fl=logger_fl)
        label_options = [lb.lower() for lb in pm.labels]
        lb2id = {lb: idx for idx, lb in enumerate(label_options)}
        eval_single = _EvalSingle(pm=pm, api_caller=ac, label_options=label_options, lb2id=lb2id, logger_fl=logger_fl)

        if concurrent:  # concurrency doesn't seem to help
            with_tqdm = dict(desc=f'Evaluating {pl.i(dnm)}', chunksize=32)
            lst = conc_map(eval_single, dset, with_tqdm=with_tqdm, mode='thread')
            preds, trues = zip(*lst)
        else:
            n_dset = len(dset)
            trues, preds = np.empty(n_dset, dtype=int), np.empty(n_dset, dtype=int)
            it = tqdm(dset, desc=f'Evaluating {pl.i(dnm)}')
            for idx, elm in enumerate(it):
                preds[idx], trues[idx] = eval_single(elm, pbar=it)

        args = dict(
            labels=[-1, *range(len(label_options))], target_names=['Label not in dataset', *label_options],
            zero_division=0, output_dict=True
        )
        report = classification_report(trues, preds, **args)
        acc = f'{report["accuracy"]:.3f}'
        logger.info(f'{pl.i(dnm)} accuracy: {pl.i(acc)}')
        logger_fl.info(f'{dnm} accuracy: {acc}')

        path = os_join(output_path, f'{dnm}.csv')
        pd.DataFrame(report).transpose().to_csv(path)


if __name__ == '__main__':
    mic.output_width = 256

    with open(os_join(u.proj_path, 'auth', 'open-ai.json')) as f:
        d = json.load(f)
        api_key, org = d['api-key'], d['organization']

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
            prompt="Say this is a test",
            max_tokens=6,
            temperature=0  # Generate w/ greedy decoding
        )
        res = requests.post(url='https://api.openai.com/v1/completions', headers=headers, json=payload)
        mic(res)
        res = json.loads(res.text)
        mic(res)
    # try_completion()

    # evaluate(model='text-ada-001', domain='in', dataset_name='emotion')
    # evaluate(model='text-davinci-002', domain='in', dataset_name='emotion')

    # evaluate(model='text-davinci-002', domain='out', dataset_name='finance_sentiment')
    # evaluate(model='text-davinci-002', domain='out', dataset_name='consumer_finance')

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
        return parser.parse_args()

    def command_prompt():
        args = parse_args()
        evaluate(model=args.model, dataset_name=args.dataset, domain=args.domain, concurrent=args.concurrent)
    command_prompt()
