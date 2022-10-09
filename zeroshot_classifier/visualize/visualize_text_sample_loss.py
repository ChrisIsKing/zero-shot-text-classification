"""
Get the text samples that models perform the worst on, and look for insights

See `zeroshot_classifier.models.binary_bert` in test mode
"""

import json
from os.path import join as os_join
from typing import List, Dict, Union, Any
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from stefutil import *
from zeroshot_classifier.util import *


def get_bad_samples(d_loss: Dict[str, np.array], k: int = 32, save: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    :param d_loss: The loss of each text sample in each dataset by a model, in iteration order
    :param k: top #samples to keep
    :return: A list of text samples with the respective loss that the model performs the worst on, sorted by performance
    :param save: Save the results to a directory path
    """
    d_out, split = dict(), 'test'
    for dnm, loss in d_loss.items():
        idxs_top = np.argpartition(loss, -k)[-k:]
        s_idxs_top = set(idxs_top)
        out = []
        for i, (txt, lbs) in enumerate(utcd.get_dataset(dnm, split).items()):
            if i in s_idxs_top:
                out.append(dict(text=txt, labels=lbs, loss=float(loss[i])))
        d_out[dnm] = sorted(out, key=lambda x: -x['loss'])
    if save:
        fnm = os_join(save, f'{now(for_path=True)}, bad_samples.json')
        with open(fnm, 'w') as fl:
            json.dump(d_out, fl, indent=4)
    return d_out


class AttentionVisualizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dataset_cache: Dict[str, Dict[str, List[str]]] = dict()

        self.model_cache = defaultdict(lambda: defaultdict(dict))  # dataset name => text => label => visualization args

        self.logger = get_logger('Binary BERT Attention Visualizer')

    def visualize(self, dataset_name: str, text: str, label: str = None, aggregate_attention: bool = True, **kwargs):
        """
        Visualize the attention weights of a text, label pair
            Intended for binary bert

        Previously computed attention weights are cached

        Should be called in a notebook only per `bertviz`
        """
        from bertviz import head_view
        split = 'test'
        if dataset_name not in self.dataset_cache:
            self.dataset_cache[dataset_name] = utcd.get_dataset(dataset_name, split)
        label_options = sconfig(f'UTCD.datasets.{dataset_name}.splits.{split}.labels')
        self.logger.info(f'Visualizing dataset {pl.i(dataset_name)} with label options {pl.i(label_options)}... ')
        if label is None:  # just assume not run on this text before
            label, args = self._get_pair(dataset_name, text, label_options)
        elif label not in self.model_cache[dataset_name][text]:
            args = self._get_pair(dataset_name, text, label)
        else:
            args = self.model_cache[dataset_name][text][label]
        self.logger.info(f'Visualizing on {pl.i(text=text, label=label)} ... ')

        if aggregate_attention:
            attn = args['attention']
            # snap batch dimension, stack by layer
            attn = torch.stack([a.squeeze() for a in attn], dim=0)  # #layer x #head x #seq_len x #seq_len
            attn = attn.mean(dim=1)  # average over all heads; L x T x T
            attn += torch.eye(attn.size(1))  # reverse residual connections; TODO: why diagonals??
            attn /= attn.sum(dim=-1, keepdim=True)  # normalize all keys for each query

            attn_res = torch.empty_like(attn)  # get recursive contribution of each token for all layers
            attn_res[0] = attn[0]
            for i in range(1, attn.size(0)):  # start from the bottom, multiply out the attentions on higher layers
                attn_res[i] = attn[i] @ attn[i - 1]
            attn_res[:, 0, 0] = 0  # ignore the score from cls to cls
            # attn_res[:, :, 1:] = 0  # keep only scores queried from cls
            args['attention'] = [a.unsqueeze(0).unsqueeze(0) for a in attn_res]
        head_view(**args, **kwargs)

    def _get_pair(self, dataset_name: str, text: str, label: Union[str, List[str]]):
        batched = isinstance(label, list)
        if batched:
            text_in, label_in = [text] * len(label), label
        else:  # single label
            text_in, label_in = [text], [label]
        tok_args = dict(padding=True, truncation='longest_first', return_tensors='pt')
        inputs = self.tokenizer(text_in, label_in, **tok_args)
        input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        attn = outputs.attentions
        if batched:
            for i, (lb, iids, tids) in enumerate(zip(label, input_ids, token_type_ids)):
                toks = self.tokenizer.convert_ids_to_tokens(iids)
                b_strt = tids.tolist().index(1)
                a = tuple(a[None, i] for a in attn)
                self.model_cache[dataset_name][text][lb] = dict(attention=a, tokens=toks, sentence_b_start=b_strt)
            scores = outputs.lg.its[:, 1]
            lb = label[scores.argmax()]  # pick the label with the highest score
            return lb, self.model_cache[dataset_name][text][lb]
        else:
            b_strt = token_type_ids[0].tolist().index(1)
            toks = self.tokenizer.convert_ids_to_tokens(input_ids[0])  # remove batch dimension
            arg = dict(attention=attn, tokens=toks, sentence_b_start=b_strt)
            self.model_cache[dataset_name][text][label] = arg  # index into the 1-element list
            return arg


if __name__ == '__main__':
    import pickle

    from stefutil import mic
    mic.output_width = 512

    model_dir_nm = os_join('binary-bert-rand-vanilla-old-shuffle-05.03.22', 'rand')
    mdl_path = os_join(u.proj_path, u.model_dir, model_dir_nm)

    def get_bad_eg():
        # dir_nm = 'in-domain, 05.09.22'
        dir_nm = 'out-of-domain, 05.10.22'
        path_eval = os_join(mdl_path, 'eval', dir_nm)
        with open(os_join(path_eval, 'eval_loss.pkl'), 'rb') as f:
            d = pickle.load(f)
        save_path = os_join(u.proj_path, 'eval', 'binary-bert', 'rand, vanilla', 'in-domain, 05.09.22')
        get_bad_samples(d, save=save_path)
    get_bad_eg()

    def visualize():
        av = AttentionVisualizer(mdl_path)
        dnm = 'emotion'
        txt = 'i feel like the writer wants me to think so and proclaiming he no longer liked pulsars is a petty and ' \
              'hilarious bit of character '
        lbl = 'anger'
        lbl = None
        av.visualize(dataset_name=dnm, text=txt, label=lbl)
    # visualize()
