import re
import math
import json
import itertools
from os.path import join as os_join
from typing import List, Tuple, Dict, Callable, Union
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from stefutil import *
from zeroshot_classifier.util.util import save_fig
from zeroshot_classifier.util.data_path import BASE_PATH, PROJ_DIR, DSET_DIR


STSb = 'stsb_multi_mt'  # Per Hugging Face
config_dict = {
    'fine-tune': dict(
        eg_sbert=dict(  # Per *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, section 4.2
            dataset_name=STSb,
            embedding_model_name='bert-base-uncased',
            max_seq_length=256,
            batch_size=16,
            n_epochs=4,
            n_eval=100,  # Total number of evaluations during training
            warmup_frac=0.1,
            pooling_model_kwargs=dict(pooling_mode_mean_tokens=True)
        )
    ),
    'datasets': {
        STSb: dict(
            n_sample=dict(
                train=5749,
                dev=1500,
                test=1379
            ),
            label_range=dict(
                min=0,
                max=5
            )
        )
    },
    'baselines': {
        'gpt2-nvidia': {
            'templates': [
                'To which category does the following document belong? : {}',
                'To which category does the following text belong? : {}',
                'To which category does the text belong? : {}',
                'To which category does the article belong? : {}',
                'How would you describe the following document? : as {}',
                'How would you describe the text? : as {}',
                'How would you describe the following text? : as {}',
                'Which best describes the text? : {}',
                'Which best describes the document? : {}',
                'Which best describes the following document? : {}',
                'Which best describes the following text? : {}',
                'The following document is _ ? : {}',
                'The following text is _ ? : {}',
                'The text is _ ? : {}',
                'The document is _ ? : {}',
                'How is the text best described? : {}',
                'How is the document best described? : {}',
                'How is the following text best described? : {}',
                'How is the following document best described? : {}',
                'Which of these choices best describes the text? : {}',
                'Which of these options best describes the text? : {}',
                'Which of these choices best describes the document? : {}',
                'Which of these options best describes the document? : {}',
                'Which of these categories best describes the following document? : {}',
                'Which of these choices best describes the following document? : {}',
                'Which of these options best describes the following text? : {}'
            ],
            'label-descriptors': dict(  # string label to natural language descriptor, as in paper
                ag_news={
                    'World': 'World News',
                    'Sports': 'Sports',
                    'Business': 'Business',
                    'Sci/Tech': 'Science & Technology'
                }
            )
        },
        'bert-mnli': dict(
            templates=dict(
                sentiment='This text expresses a {} sentiment',
                intent='This text expresses the intent of {}',
                topic='This text belongs to the topic of {}'
            )
        ),
    },
    'UTCD': dict(
        datasets=dict(
            # in-domain evaluation has the same labels as training
            go_emotion=dict(
                path='UTCD/in-domain/go_emotion', aspect='sentiment', eval_labels_same=True, domain='in',
                name='GoEmotions', name_compact='GoEmotions'
            ),
            sentiment_tweets_2020=dict(
                path='UTCD/in-domain/sentiment_tweets_2020', aspect='sentiment', eval_labels_same=True, domain='in',
                name='TweetEval', name_compact='TweetEval'
            ),
            emotion=dict(
                path='UTCD/in-domain/emotion', aspect='sentiment', eval_labels_same=True, domain='in',
                name='Emotion', name_compact='Emotion'
            ),
            # not `eval_labels_same` := has some unique test labels
            sgd=dict(
                path='UTCD/in-domain/sgd', aspect='intent', eval_labels_same=False, domain='in',
                name='Schema-Guided Dialogue', name_compact='SGD'
            ),
            clinc_150=dict(
                path='UTCD/in-domain/clinc_150', aspect='intent', eval_labels_same=True, domain='in',
                name='Clinc-150', name_compact='Clinc-150'
            ),
            slurp=dict(
                path='UTCD/in-domain/slurp', aspect='intent', eval_labels_same=False, domain='in',
                name='SLURP', name_compact='SLURP'
            ),
            ag_news=dict(
                path='UTCD/in-domain/ag_news', aspect='topic', eval_labels_same=True, domain='in',
                name='AG News', name_compact='AG News'
            ),
            dbpedia=dict(
                path='UTCD/in-domain/dbpedia', aspect='topic', eval_labels_same=True, domain='in',
                name='DBpedia', name_compact='DBpedia'
            ),
            yahoo=dict(
                path='UTCD/in-domain/yahoo', aspect='topic', eval_labels_same=True, domain='in',
                name='Yahoo Answer Topics', name_compact='Yahoo'
            ),
            # Out-of-domain datasets: only test split used & intended for evaluation
            amazon_polarity=dict(
                path='UTCD/out-of-domain/amazon_polarity', aspect='sentiment', eval_labels_same=True, domain='out',
                name='Amazon Review Polarity', name_compact='Amazon Polarity'
            ),
            finance_sentiment=dict(
                path='UTCD/out-of-domain/finance_sentiment', aspect='sentiment', eval_labels_same=True, domain='out',
                name='Financial Phrase Bank', name_compact='Fin. Phrase Bank'
            ),
            yelp=dict(
                path='UTCD/out-of-domain/yelp', aspect='sentiment', eval_labels_same=True, domain='out',
                name='Yelp Review', name_compact='Yelp'
            ),
            banking77=dict(
                path='UTCD/out-of-domain/banking77', aspect='intent', eval_labels_same=True, domain='out',
                name='Banking77', name_compact='Banking77'
            ),
            snips=dict(
                path='UTCD/out-of-domain/snips', aspect='intent', eval_labels_same=True, domain='out',
                name='SNIPS', name_compact='SNIPS'
            ),
            nlu_evaluation=dict(
                path='UTCD/out-of-domain/nlu_evaluation', aspect='intent', eval_labels_same=True, domain='out',
                name='NLU Evaluation', name_compact='NLU Eval'
            ),
            # Removed for too many options, blowing up GPT2's 1024 token length
            # arxiv=dict(path='UTCD/out-of-domain/arxiv', aspect='topic', eval_labels_same=True, domain='out'),
            multi_eurlex=dict(
                path='UTCD/out-of-domain/multi_eurlex', aspect='topic', eval_labels_same=True, domain='out',
                name='MultiEURLEX', name_compact='MultiEURLEX'
            ),
            patent=dict(
                path='UTCD/out-of-domain/patent', aspect='topic', eval_labels_same=True, domain='out',
                name='Big Patent', name_compact='Patent'
            ),
            consumer_finance=dict(
                path='UTCD/out-of-domain/consumer_finance', aspect='topic', eval_labels_same=True, domain='out',
                name='Consumer Finance Complaints', name_compact='Consumer Finance'
            )
        ),
        aspects=['sentiment', 'intent', 'topic'],
        domains=['in', 'out'],
        num_aspect=3,
        num_dataset_per_aspect=6,
        num_dataset_per_domain_per_aspect=3,
        num_domain=2,
        num_dataset_per_domain=9,
        dataset_ext='json'  # all in json
    ),
    'training': {
        'implicit-on-text': {
            'encode-aspect': {
                'aspect2aspect-token': dict(sentiment='<|sentiment|>', intent='<|intent|>', topic='<|topic|>')
            },
            'encode-sep': {'aspect-sep-token': '<|ASPECT-SEP|>'}
        },
        'strategies': [
            'vanilla',
            'implicit',  # prepend aspect text before each label
            'implicit-on-text-encode-aspect',  # encode each of the 3 aspects as 3 special tokens, followed by text
            'implicit-on-text-encode-sep',  # encode aspects normally, but add special token between aspect and text
            'explicit'  # see `zeroshot_classifier.explicit.binary_bert.py` for explicit training
        ]
    },
    'random-seed': 77,
    'check-arg': [
        dict(
            display_name='Model Name', attr_name='model_name',
            accepted_values=[
                'bert-seq-cls',  # not a Zeroshot framework, a supervised learning upperbound
                'binary-bert', 'bert-nli', 'bi-encoder', 'dual-bi-encoder', 'gpt2-nvidia'
            ]
        ),
        dict(
            display_name='Dataset Domain', attr_name='dataset_domain',
            accepted_values=['in', 'out']
        ),
        dict(
            display_name='Sampling Strategy', attr_name='sampling_strategy',
            accepted_values=['rand', 'vect', 'none', 'NA']
        ),
        dict(
            display_name='Training strategy', attr_name='training_strategy',
            accepted_values=[
                'vanilla',
                'implicit',
                'implicit-on-text-encode-aspect',
                'implicit-on-text-encode-sep',
                'explicit'
            ]
        ),
        dict(
            display_name='GPT2 Training Strategy', attr_name='gpt2_training_strategy',
            accepted_values=['vanilla', 'implicit', 'explicit']
        )
    ]
}

path_dset = os_join(BASE_PATH, PROJ_DIR, DSET_DIR)
ext = config_dict['UTCD']['dataset_ext']


def _re_call() -> Callable[[str], int]:
    if not hasattr(_re_call, 'token_pattern'):
        # taken from sklearn.CountVectorizer
        _re_call.token_pattern = re.compile(r'(?u)\b\w+\b')
    return lambda x: len(_re_call.token_pattern.findall(x))


def _hf_call(model_name) -> Callable[[Union[str, List[str]]], Union[int, List[int]]]:
    if not hasattr(_hf_call, 'd'):
        _hf_call.d = {}
    d = _hf_call.d
    if model_name not in d:
        d[model_name] = AutoTokenizer.from_pretrained(model_name)

    def _call(x):
        ids = d[model_name](x)['input_ids']
        if isinstance(x, str):
            return len(ids)
        else:
            return [len(i) for i in ids]
    return _call


def get_tokenizer_len(s: Union[str, List[str]], mode: str = 're') -> Union[int, List[int]]:
    assert mode in ['re', 'bert', 'gpt2']
    if not hasattr(get_tokenizer_len, 'd_f'):
        get_tokenizer_len.d_f = dict(
            re=_re_call(),
            bert=_hf_call('bert-base-cased'),
            gpt2=_hf_call('gpt2')
        )
    return get_tokenizer_len.d_f[mode](s)


tokenize_modes = ['re', 'bert', 'gpt2']


def path2dataset_info(d: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    :return: 3-tuple of (
        dataset label information per split for `config`,
        dataset token information per dataset for `config`,
        number of tokens for plot
    )
    """
    path = os_join(path_dset, f'{d["path"]}.{ext}')
    with open(path) as fl:
        dsets: Dict = json.load(fl)

    def split2info(split, dset: Dict[str, List[str]], count_token_length: bool = True) -> Dict:
        # Based on heuristics on how the `json` are stored
        # creating a list of all the strings consume memory for prohibitively large datasets
        n_text_, n_pair_ = len(dset.keys()), sum([len(lbs) for lbs in dset.values()])
        lbs_uniq = set().union(*dset.values())
        n_multi_label = sum([len(lbs_) > 1 for lbs_ in dset.values()])
        txt_n_toks, lb_n_toks = None, None
        if count_token_length:
            txt_n_toks, lb_n_toks = dict(), dict()
            for mode in tokenize_modes:
                n, desc_t, desc_l = 16, f'{split}-{mode}-text', f'{split}-{mode}-label'
                lb2tokenize_len = {lb: get_tokenizer_len(lb, mode) for lb in lbs_uniq}

                counter_txt, counter_lb = Counter(), Counter()
                if mode == 're':
                    for t in tqdm(dset.keys(), total=len(dset), desc=f'{desc_t:>{n}}'):
                        counter_txt[get_tokenizer_len(t, mode)] += 1
                else:
                    batch_size = 2048*2
                    for grp in tqdm(
                            group_n(dset.keys(), batch_size),
                            total=math.ceil(len(dset) / batch_size), desc=f'{desc_t:>{n}}'
                    ):
                        lens: List[int] = get_tokenizer_len(list(grp), mode)
                        counter_txt.update(lens)
                for t in tqdm(dset.values(), desc=f'{desc_l:>{n}}'):
                    for lb in t:
                        counter_lb[lb2tokenize_len[lb]] += 1
                txt_n_toks[mode], lb_n_toks[mode] = counter_txt, counter_lb
        return dict(
            labels=sorted(lbs_uniq),
            n_label=len(lbs_uniq),
            n_text=n_text_,
            n_pair=n_pair_,
            multi_label=n_text_ < n_pair_,
            n_multi_label=n_multi_label,
            txt_n_toks=txt_n_toks,
            lb_n_toks=lb_n_toks
        )
    labels, aspect = dsets.pop('labels'), dsets.pop('aspect')
    assert aspect == d['aspect']
    d_out = {  # ignore out of domain train split for potentially too large
        split: split2info(split, dset, count_token_length=not (split == 'train' and d['domain'] == 'out'))
        for split, dset in dsets.items()
    }  # Labels for each split
    assert all(split in ['train', 'test'] for split in d_out.keys())
    # sum over all splits of the dataset for token length computation
    txt_n_toks_all = [d_out.pop('txt_n_toks') for d_out in d_out.values()]
    lb_n_toks_all = [d_out.pop('lb_n_toks') for d_out in d_out.values()]
    txt_n_toks_all = [e for e in txt_n_toks_all if e]  # pop from the dict, then remove them for stats
    lb_n_toks_all = [e for e in lb_n_toks_all if e]
    txt_n_toks_all = {mode: sum([c[mode] for c in txt_n_toks_all], start=Counter()) for mode in tokenize_modes}
    lb_n_toks_all = {mode: sum([c[mode] for c in lb_n_toks_all], start=Counter()) for mode in tokenize_modes}

    def counter2mean(c: Counter) -> float:
        lens, counts = zip(*c.items())
        return np.average(lens, weights=counts)
    avg_toks = {
        **{f'{mode}-txt_avg_tokens': counter2mean(txt_n_toks_all[mode]) for mode in tokenize_modes},
        **{f'{mode}-lb_avg_tokens': counter2mean(lb_n_toks_all[mode]) for mode in tokenize_modes}
    }
    assert set(labels) == set().union(*[set(d['labels']) for d in d_out.values()])
    if d['eval_labels_same']:
        assert d_out['train']['labels'] == d_out['test']['labels']
    return d_out, avg_toks, dict(text=txt_n_toks_all, label=lb_n_toks_all)


def extract_utcd_meta() -> Dict:
    d_dsets: Dict = config_dict['UTCD']['datasets']
    logger = get_logger('Process UTCD')
    d_n_toks = dict()
    for dnm, d_dset in d_dsets.items():
        logger.info(f'Processing {pl.i(dnm)}... ')
        d_meta, d_avg_tok, d_n_toks[dnm] = path2dataset_info(d_dset)
        d_dset['splits'] = d_meta
        d_dset.update(d_avg_tok)
    dnms = sorted(d_dsets)  # All datasets, in- and out-of-domain, share the same dataset <=> id mapping
    config_dict['UTCD']['dataset_id2name'] = dnms
    config_dict['UTCD']['dataset_name2id'] = {dnm: i for i, dnm in enumerate(dnms)}
    return d_n_toks


def plot_utcd_n_toks(d_n_toks: Dict, domain: str, save=True):
    def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
        # Credit: https://stackoverflow.com/a/29677616/10732321
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
            'quantiles should be in [0, 1]'

        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)
    logger = get_logger('Token Lengths Distribution Plot')
    d_df = dict()
    text_types = ['text', 'label']
    for text_type, mode in itertools.product(text_types, tokenize_modes):
        logger.info(f'Processing {pl.i(text_type)} with {pl.i(mode)} tokenization')

        def dnm2dset(dnm: str) -> List[Tuple[int, int, str]]:
            counter = d_n_toks[dnm][text_type][mode]
            lens, counts = zip(*counter.items())
            return [(l, c, dnm) for l, c in zip(lens, counts)]
        toks_unrolled = sum([dnm2dset(dnm) for dnm in d_n_toks.keys()], start=[])
        # `count` is a pd.DataFrame specific attribute
        d_df[(text_type, mode)] = pd.DataFrame(toks_unrolled, columns=['n_token', 'counts', 'dataset_name'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    n_tt, n_tm = len(text_types), len(tokenize_modes)
    for i_row, i_col in itertools.product(range(n_tt), range(n_tm)):
        text_type, mode = text_types[i_row], tokenize_modes[i_col]
        logger.info(f'Plotting {pl.i(text_type)} with {pl.i(mode)} tokenization')
        ax = axes[i_row, i_col]
        df = d_df[(text_type, mode)]
        legend = i_row == 0 and i_col == 0
        sns.histplot(
            data=df, x='n_token', hue='dataset_name', weights='counts',
            kde=text_type == 'text', kde_kws=dict(gridsize=2048), discrete=True, common_norm=False, stat='density',
            palette='husl', legend=legend, ax=ax
        )
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'{text_type} with {mode} tokenization')
        if text_type == 'text':  # empirical, cos there are outliers for `text`s
            p = norm().cdf(3)  # quantile at 3std
            mi = df.n_token.min()
            ma = round(weighted_quantile(df.n_token, [p], sample_weight=df.counts)[0])
            ax.set_xlim([mi, ma])
        else:
            xticks = ax.get_xticks()  # enforce integer ticks
            ax.set_xticks(list(range(math.floor(xticks.min()), math.ceil(xticks.max()) + 1)))
    domain = 'in-domain' if domain == 'in' else 'out-of-domain eval'
    title = f'Tokenization length distribution across {domain} datasets'
    plt.suptitle(title)
    fig.supxlabel('#token')
    fig.supylabel('Density')
    if save:
        save_fig(title)
    else:
        plt.show()


d_n_tok = extract_utcd_meta()
for dom in ['in']:
    # plot only in-domain data as out-of-domain tokens lengths are too long,
    # resulting in prohibitively large # of patches for bar-plot to terminate soon
    d_n_tok_ = {dnm: v for dnm, v in d_n_tok.items() if get(config_dict, f'UTCD.datasets.{dnm}.domain') == dom}
    plot_utcd_n_toks(d_n_tok_, domain=dom, save=True)


if __name__ == '__main__':
    from zeroshot_classifier.util.data_path import PKG_NM

    fl_nm = 'config.json'
    mic(config_dict)
    with open(os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
