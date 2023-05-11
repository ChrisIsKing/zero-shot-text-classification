import re
import os
import json
import time
import logging
from os.path import join as os_join
from typing import List, Dict, Any, Union, Optional
from argparse import ArgumentParser
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import datasets
import openai
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tenacity import retry, wait_random_exponential

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

    def __init__(self, model: str = 'text-ada-001', batched: bool = False, delay: float = None):
        self.model = model
        self.batched = batched

        self.delay = delay

    @retry(wait=wait_random_exponential(min=1, max=60 * 30))  # Wait for 30min
    def completion(self, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        return openai.Completion.create(**kwargs)

    def __call__(self, prompt: Union[str, List], rand_sleep: bool = True, **kwargs) -> Union[str, List[str]]:
        if self.batched:
            assert isinstance(prompt, list)
        else:
            assert isinstance(prompt, str)

        payload = dict(
            model=self.model,
            temperature=0,  # Generate w/ greedy decoding
            stop='\n',
            max_tokens=32
        )
        payload['prompt'] = prompt
        payload.update(kwargs)

        res = self.completion(**payload)
        res = res.choices
        if self.batched:
            assert len(res) == len(prompt)
            ret = [''] * len(prompt)
            for e in res:
                ret[e.index] = e.text
            return ret
        else:
            assert len(res) == 1
            res = res[0]
            return res.text


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
        # return f'Text: {text}\nQuestion: {question}\n Answer:'  # TODO: This template works w/ `davinci` better?
        return f'{question}\n Text: {truncate_text(text)} \n Answer:'  # This template works w/ `curie` better


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
        self.batched = api_caller.batched

        self.label_options = label_options
        self.lb2id = lb2id

        self.logger_fl = logger_fl

        self.return_text = return_text

    def __call__(self, e: Union[Dict[str, Any], List[Dict[str, Any]]], pbar=None) -> Union[_EvalSingleOut, List[_EvalSingleOut]]:
        if self.batched:
            d: List[Dict]
            lst_txt, lst_lbs = [i['text'] for i in e], [i['labels'] for i in e]
            assert isinstance(lst_txt[0], str) and isinstance(lst_lbs[0], list)  # sanity check
            prompts = [self.pm(txt) for txt in lst_txt]
            res = self.ac(prompts)

            ret = []
            for txt, lbs, ppt, a in zip(lst_txt, lst_lbs, prompts, res):
                ret.append(self._ret_single(text=txt, labels=lbs, prompt=ppt, answer=a, pbar=pbar))
            return ret
        else:
            txt, lbs = e['text'], e['labels']
            prompt = self.pm(txt)
            answer = self.ac(prompt)
            return self._ret_single(text=txt, labels=lbs, prompt=prompt, answer=answer, pbar=pbar)

    def _ret_single(
            self, text: str = None, labels: List[int] = None, prompt: str = None, answer: str = None, pbar=None
    ) -> _EvalSingleOut:
        if pbar:
            _d_log = dict(labels=[self.label_options[i] for i in labels], answer=[answer])
            pbar.set_postfix({k: pl.i(v) for k, v in _d_log.items()})
        answer = answer.lower().strip()  # TODO: maybe GPT3 generates multiple answers?

        ret: Dict[str, Any]
        if answer in self.label_options:
            ret = dict(pred=self.lb2id[answer], true=self.lb2id[answer])
        else:
            logger.warning(f'Generated {pl.i([answer])}, not one of label options')
            self.logger_fl.warning(f'Generated {pl.nc([answer])}, not one of label options')

            ret = dict(pred=-1, true=labels[0])
        if self.return_text:
            ret.update(meta=GPT3EvalMeta(text=text, prompt=prompt, generated=answer))
        return _EvalSingleOut(**ret)


def evaluate(
        model: str = 'text-ada-001', domain: str = 'in', dataset_name: str = 'all', concurrent: bool = False,
        batched: Union[bool, int] = False, delay: float = None,
        subsample: Union[bool, int] = False, subsample_seed: int = 77, store_meta: bool = False,
        store_frequency: Optional[int] = None, resume: List[str] = None
):
    ac = ApiCaller(model=model, batched=batched, delay=delay)

    if dataset_name == 'all' and subsample:
        raise NotImplementedError('Subsampling intended for single dataset')
    dataset_names = utcd_util.get_eval_dataset_names(domain=domain, dataset_name=dataset_name)

    _preds, _trues, _infs = None, None, []
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
    d_log: Dict[str, Any] = dict(
        model_name=model, batched=batched, delay=delay, domain=domain, dataset_names=dataset_names,
        output_path=output_path,
        concurrent=concurrent, subsample=subsample, subsample_seed=subsample_seed, store_meta=store_meta
    )
    d_log['store_frequency'] = store_frequency
    logger.info(f'Evaluating GPT3 model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating GPT3 model w/ {d_log}... ')

    os.makedirs(output_path, exist_ok=True)

    for dnm in dataset_names:
        n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.test.n_text')
        n_tgt = 5000 if isinstance(subsample, bool) else subsample

        if subsample and n_txt > n_tgt:
            dset = utcd_util.subsample_dataset(dataset_name=dnm, split='test', n_tgt=n_tgt, seed=subsample_seed)
        else:
            dset = get_dataset(dnm, splits='test')['test']
        dset: datasets.Dataset
        n_dset_total = len(dset)
        n_dset_remain = n_dset_total

        if resume:
            n_ori = len(dset)
            ran_txts = set(e['text'] for e in _infs)

            def filt(sample: Dict[str, Any]) -> bool:
                return sample['text'] not in ran_txts
            dset = dset.filter(filt)
            # sanity check, every text completed should be accounted for, exactly once
            n_dset_remain = len(dset)
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
            n_completed = len(lst_meta)
            if n_completed % store_frequency == 0 or n_completed == n_dset_remain:
                logger.info(f'Writing eval instances to {pl.i(meta_path)}...')
                logger_fl.info(f'Writing eval instances to {meta_path}...')
                d_ = d_log
                infs = [asdict(m) for m in lst_meta]
                if resume:
                    infs = _infs + infs
                if concurrent:  # a numpy array created
                    __preds, __trues = preds, trues
                    total_completed = len(__preds)
                else:
                    total_completed = len(_infs) + n_completed
                    __preds = preds[:total_completed].tolist()
                    __trues = trues[:total_completed].tolist()
                assert len(__preds) == len(__trues) == len(infs)  # sanity check
                d_['#completed'] = total_completed
                d_.update(dict(inferences=infs, preds=__preds, trues=__trues))

                with open(meta_path, 'w') as f_:
                    json.dump(d_, f_, indent=4)

        bsz = None
        if batched:
            bsz = 8 if isinstance(batched, bool) else batched

        with_tqdm = dict(desc=f'Evaluating {pl.i(dnm)}', total=n_dset_remain)
        if concurrent:  # TODO: concurrent batched
            preds, trues = (_preds, _trues) if resume else ([], [])
            # order irrelevant
            for e in conc_yield(eval_single, dset, with_tqdm=with_tqdm, mode='thread', n_worker=4):
                preds.append(e.pred)
                trues.append(e.true)
                if store_meta:
                    lst_meta.append(e.meta)
                    write_meta()
        else:
            trues, preds = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)
            if resume:
                trues[:len(_trues)] = _trues
                preds[:len(_preds)] = _preds
            if batched:
                with tqdm(**with_tqdm) as pbar:
                    idx = 0
                    for elms in DataLoader(dset, batch_size=bsz, shuffle=False, collate_fn=lambda x: x):
                        for e in eval_single(elms):
                            preds[idx] = e.pred
                            trues[idx] = e.true
                            if store_meta:
                                lst_meta.append(e.meta)
                                write_meta()
                            pbar.update(1)
                            idx += 1
                assert idx == n_dset_remain  # sanity check
            else:
                it = tqdm(dset, **with_tqdm)
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

        csv_path = os_join(output_path, f'{dnm}.csv')
        pd.DataFrame(report).transpose().to_csv(csv_path)
        acc = f'{report["accuracy"]:.3f}'
        logger.info(f'{pl.i(dnm)} accuracy: {pl.i(acc)}')
        logger_fl.info(f'{dnm} accuracy: {acc}')


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


if __name__ == '__main__':
    mic.output_width = 256

    with open(os_join(u.proj_path, 'auth', 'open-ai.json')) as f:
        auth = json.load(f)
        org = auth['organization']
        api_key = auth['api-key']
    openai.api_key = api_key

    # evaluate(model='text-ada-001', domain='in', dataset_name='emotion')
    # evaluate(model='text-curie-001', domain='out', dataset_name='multi_eurlex', concurrent=True)
    # evaluate(model='text-curie-001', domain='in', dataset_name='emotion', concurrent=True)
    # evaluate(model='text-curie-001', domain='in', dataset_name='finance_sentiment', concurrent=True)
    # evaluate(model='text-curie-001', domain='in', dataset_name='banking77', concurrent=True, subsample=True)
    # evaluate(model='text-davinci-002', domain='in', dataset_name='emotion')
    # evaluate(model='text-davinci-002', domain='out', dataset_name='finance_sentiment')
    # evaluate(model='text-davinci-002', domain='out', dataset_name='consumer_finance')
    # evaluate(model='text-davinci-002', domain='out', dataset_name='amazon_polarity', concurrent=True)

    run_args = dict(model='text-curie-001', subsample=True, store_meta=True, store_frequency=10)
    # dnm = 'amazon_polarity'
    # dnm = 'yelp'
    # dnm_ = 'consumer_finance'
    # dnm_ = 'slurp'
    dnm_ = 'multi_eurlex'  # TODO: doesn't work w/ batched requests???
    # dnm_ = 'sgd'
    rsm = [os_join(
        u.eval_path, '2022-12-04_17-34-01_Zeroshot-GPT3-Eval_{md=text-curie-001, dm=out, dnm=multi_eurlex}',
        '22-12-04_out-of-domain', f'{dnm_}_meta.json'
    )]
    # evaluate(domain='out', dataset_name=dnm_, **run_args, concurrent=True, delay=12, resume=rsm)

    def command_prompt():
        args = parse_args()
        evaluate(
            model=args.model, dataset_name=args.dataset, domain=args.domain, concurrent=args.concurrent,
            subsample=args.subsample, subsample_seed=args.subsample_seed
        )
    command_prompt()
