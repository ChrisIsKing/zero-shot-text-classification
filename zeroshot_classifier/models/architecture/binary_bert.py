from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

from stefutil import *


def load_sliced_binary_bert(
        model_name: str = 'bert-base-uncased', max_position_embeddings: int = 512
) -> Tuple[BertTokenizer, nn.Module]:
    """
    :param model_name: A hugging face model name
    :param max_position_embeddings: Max model token size

    Intended for loading a pretrained 512-token BERT model,
        with smaller max token length by chopping off later positional embeddings
    """
    conf = BertConfig.from_pretrained(model_name)
    n_tok_ori = conf.max_position_embeddings
    assert max_position_embeddings < n_tok_ori, \
        f'Intended for a {pl.i("max_position_embeddings")} smaller than original model size of {pl.i(n_tok_ori)}, ' \
        f'but got {pl.i(max_position_embeddings)}'
    conf.max_position_embeddings = max_position_embeddings
    tokenizer = BertTokenizer.from_pretrained(model_name, model_max_length=max_position_embeddings)
    model = BertForSequenceClassification.from_pretrained(model_name, config=conf, ignore_mismatched_sizes=True)
    # Should observe 2 warnings, one expected warning for initializing BertSeqCls from pre-trained Bert
    # One is for the mismatched position embedding

    # for overriding the positional embedding; Another SeqCls warning here
    model_dummy = BertForSequenceClassification.from_pretrained(model_name)
    state_d = model_dummy.bert.embeddings.position_embeddings.state_dict()
    assert set(state_d.keys()) == {'weight'}  # sanity check
    weight_pretrained = state_d['weight']
    assert weight_pretrained.shape == (n_tok_ori, conf.hidden_size)
    del model_dummy
    del state_d

    with torch.no_grad():
        # Keep the first tokens
        model.bert.embeddings.position_embeddings.weight[:] = weight_pretrained[:max_position_embeddings]
    return tokenizer, model


if __name__ == '__main__':
    load_sliced_binary_bert(max_position_embeddings=256)
