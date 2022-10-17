import re
import os
import json
import time
from typing import Dict, Any

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
            time.sleep(1)
            res = _call()

        res = json.loads(res.text)
        # mic(res)
        assert len(res['choices']) == 1  # sanity check only generated one completion
        res = res['choices'][0]
        assert res['finish_reason'] == 'stop'  # TODO: make sure GPT3 generates all it wants to
        return res['text']


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

    def __init__(self, dataset_name: str = None, max_text_length: int = 1024, max_prompt_length: int = 1024 + 256):
        self.dataset_name = dataset_name
        self.labels = sconfig(f'UTCD.datasets.{dataset_name}.splits.test.labels')  # Take labels from the test split

        self.n_cls = len(self.labels)

        self.max_text_length = max_text_length
        self.max_prompt_length = max_prompt_length

        d_log = {
            'dataset_name': dataset_name, 'labels': self.labels, '#class': self.n_cls,
            'max_text_length': max_text_length, 'max_prompt_length': max_prompt_length
        }
        PromptMap.logger.info(f'Prompt Map initialized with: {pl.i(d_log)}')

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
                                     f'{n_prompt} -> {self.max_prompt_length}')
            prompt = self._to_prompt(question=question, text=text)
        return prompt

    @staticmethod
    def _to_prompt(question: str = None, text: str = None):
        # return f'Text: {text}\nQuestion: {question}\n Answer:'
        return f'{question}\n Text: {truncate_text(text)} \n Answer:'


def evaluate(model: str = 'text-ada-001', domain: str = 'in'):
    ac = ApiCaller(model=model)

    output_dir_nm = f'{now(for_path=True)}_Zeroshot-GPT3-{model}'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))
    os.makedirs(output_path, exist_ok=True)

    dataset_names = utcd_util.get_dataset_names(domain)
    log_fnm = f'{now(for_path=True)}_GPT3_{model}_{domain}_Eval'
    logger_fl = get_logger('GPT3 Eval', typ='file-write', file_path=os_join(output_path, f'{log_fnm}.log'))

    d_log = dict(model_name=model, domain=domain, output_path=output_path)
    logger.info(f'Evaluating GPT3 model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating GPT3 model w/ {d_log}... ')

    for dnm in dataset_names:
        if dnm != 'emotion':  # TODO: debugging
            continue
        dset = get_dataset(dnm, splits='test')['test']
        pm = PromptMap(dataset_name=dnm)
        label_options = [lb.lower() for lb in pm.labels]
        lb2id = {lb: idx for idx, lb in enumerate(label_options)}

        n_dset = len(dset)
        trues, preds = np.empty(n_dset, dtype=int), np.empty(n_dset, dtype=int)

        def _call(e: Dict[str, Any]):
            txt, lbs = e['text'], e['labels']
            prompt = pm(txt)
            answer = ac(prompt)  # TODO: maybe GPT3 generates multiple answers?
            answer = answer.lower()

            if answer not in label_options:
                logger_fl.warning(f'Generated {pl.i(answer)}, not one of label options')
                return -1, lbs[0]
            else:
                return lb2id[answer], lb2id[answer]

        concurrent = False
        if concurrent:  # concurrency doesn't seem to help
            conc_map(_call, dset, with_tqdm=dict(desc=f'Evaluating {pl.i(dnm)}', chunksize=32), mode='process')
        else:
            it = tqdm(dset, desc=f'Evaluating {pl.i(dnm)}')
            for idx, elm in enumerate(it):
                preds[idx], trues[idx] = _call(elm)

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
        exit(1)


if __name__ == '__main__':
    mic.output_width = 128

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

    evaluate(model='text-ada-001', domain='in')
    # evaluate(model='text-davinci-002', domain='in')
