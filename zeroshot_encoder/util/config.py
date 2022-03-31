from collections import Counter

from scipy.stats import norm
from transformers import AutoTokenizer
from tqdm import tqdm

from zeroshot_encoder.util import *


STSb = 'stsb_multi_mt'  # Per Hugging Face
config = {
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
            emotion=dict(
                path='UTCD/in-domain/emotion', aspect='sentiment', eval_labels_same=True, domain='in'),
            go_emotion=dict(
                path='UTCD/in-domain/go_emotion', aspect='sentiment', eval_labels_same=True, domain='in'),
            sentiment_tweets_2020=dict(
                path='UTCD/in-domain/sentiment_tweets_2020', aspect='sentiment', eval_labels_same=True, domain='in'
            ),
            clinc_150=dict(
                path='UTCD/in-domain/clinc_150', aspect='intent', eval_labels_same=True, domain='in'),
            # `eval_labels_same` := has some unique test labels
            sgd=dict(path='UTCD/in-domain/sgd', aspect='intent', eval_labels_same=False, domain='in'),
            slurp=dict(path='UTCD/in-domain/slurp', aspect='intent', eval_labels_same=False, domain='in'),
            ag_news=dict(path='UTCD/in-domain/ag_news', aspect='topic', eval_labels_same=True, domain='in'),
            dbpedia=dict(path='UTCD/in-domain/dbpedia', aspect='topic', eval_labels_same=True, domain='in'),
            yahoo=dict(path='UTCD/in-domain/yahoo', aspect='topic', eval_labels_same=True, domain='in'),
            # Out-of-domain datasets: test split intended to evaluation
            # TODO: until new multi-label format supported
            amazon_polarity=dict(
                path='UTCD-ood/amazon_polarity', aspect='sentiment', eval_labels_same=True, domain='out'
            ),
            finance_sentiment=dict(
                path='UTCD-ood/finance_sentiment', aspect='sentiment', eval_labels_same=True, domain='out'
            ),
            yelp=dict(path='UTCD-ood/yelp', aspect='sentiment', eval_labels_same=True, domain='out'),
            # Removed for too many options blow up GPT2's 1024 token length; TODO: remove, keep now cos plotting
            arxiv=dict(path='UTCD-ood/arxiv', aspect='topic', eval_labels_same=True, domain='out'),
            multi_eurlex=dict(
              path='UTCD-ood/multi_eurlex', aspect='topic', eval_labels_same=True, domain='out'),
            patent=dict(path='UTCD-ood/patent', aspect='topic', eval_labels_same=True, domain='out'),
            consumer_finance=dict(
                path='UTCD-ood/consumer_finance', aspect='topic', eval_labels_same=True, domain='out'
            ),
            banking77=dict(path='UTCD-ood/banking77', aspect='intent', eval_labels_same=True, domain='out'),
            snips=dict(path='UTCD-ood/snips', aspect='intent', eval_labels_same=True, domain='out'),
            nlu_evaluation=dict(
                path='UTCD-ood/nlu_evaluation', aspect='intent', eval_labels_same=True, domain='out'
            )
        ),
        dataset_ext='json'  # all in json
    ),
    'random-seed': 77
}

path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
ext = config['UTCD']['dataset_ext']


def _re_call() -> Callable[[str], int]:
    if not hasattr(_re_call, 'token_pattern'):
        # taken from sklearn.CountVectorizer, which was `r"(?u)\b\w\w+\b"`
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
    path = os.path.join(path_dset, f'{d["path"]}.{ext}')
    with open(path) as fl:
        dsets: Dict = json.load(fl)

    def split2info(split, dset: Dict[str, List[str]]) -> Dict:
        # Based on heuristics on how the `json` are stored
        # creating a list of all the strings consume memory for prohibitively large datasets
        n_text_, n_pair_ = len(dset.keys()), sum([len(lbs) for lbs in dset.values()])
        lbs_uniq = set().union(*dset.values())
        n_multi_label = sum([len(lbs_) > 1 for lbs_ in dset.values()])
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
                        group_n(dset.keys(), batch_size), total=math.ceil(len(dset) / batch_size), desc=f'{desc_t:>{n}}'
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
    d_out = {split: split2info(split, dset) for split, dset in dsets.items()}  # Labels for each split
    assert all(split in ['train', 'test'] for split in d_out.keys())
    # sum over all splits of the dataset for token length computation
    txt_n_toks_all = [d_out.pop('txt_n_toks') for d_out in d_out.values()]
    lb_n_toks_all = [d_out.pop('lb_n_toks') for d_out in d_out.values()]
    txt_n_toks_all = {mode: sum([c[mode] for c in txt_n_toks_all], start=Counter()) for mode in tokenize_modes}
    lb_n_toks_all = {mode: sum([c[mode] for c in lb_n_toks_all], start=Counter()) for mode in tokenize_modes}

    def counter2mean(c: Counter) -> float:
        lens, counts = zip(*c.items())
        return np.average(lens, weights=counts)
    avg_toks = {f'{mode}-txt_avg_tokens': counter2mean(txt_n_toks_all[mode]) for mode in tokenize_modes} | \
               {f'{mode}-lb_avg_tokens': counter2mean(lb_n_toks_all[mode]) for mode in tokenize_modes}
    assert set(labels) == set().union(*[set(d['labels']) for d in d_out.values()])
    if d['eval_labels_same']:
        assert d_out['train']['labels'] == d_out['test']['labels']
    return d_out, avg_toks, dict(text=txt_n_toks_all, label=lb_n_toks_all)


def extract_utcd_meta() -> Dict:
    d_dsets: Dict = config['UTCD']['datasets']
    logger = get_logger('Process UTCD')
    d_n_toks = dict()
    for dnm, d_dset in d_dsets.items():
        logger.info(f'Processing {logi(dnm)}... ')
        if d_dset['domain'] == 'in':
            d_meta, d_avg_tok, d_n_toks[dnm] = path2dataset_info(d_dset)
            d_dset['splits'] = d_meta
            d_dset.update(d_avg_tok)
    dnms = sorted(d_dsets)  # All datasets, in- and out-of-domain, share the same dataset <=> id mapping
    config['UTCD']['dataset_name2id'] = {dnm: i for i, dnm in enumerate(dnms)}
    config['UTCD']['dataset_id2name'] = {i: dnm for i, dnm in enumerate(dnms)}

    return d_n_toks


def plot_utcd_n_toks(d_n_toks: Dict, save=True):
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
        logger.info(f'Processing {logi(text_type)} with {logi(mode)} tokenization')

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
        logger.info(f'Plotting {logi(text_type)} with {logi(mode)} tokenization')
        ax = axes[i_row, i_col]
        df = d_df[(text_type, mode)]
        legend = i_row == 0 and i_col == 0
        sns.histplot(
            data=df, x='n_token', hue='dataset_name', kde=text_type == 'text', discrete=True, weights='counts',
            palette='husl',
            legend=legend, common_norm=False, ax=ax, stat='density'
        )
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'{text_type} with {mode} tokenization')
        if text_type == 'text':  # empirical, cos there are outliers for `text`s
            p = norm().cdf(3)  # quantile at 3std
            mi, ma = df.n_token.min(), math.ceil(df.n_token.quantile(p))
            ma = weighted_quantile(df.n_token, [p], sample_weight=df.counts)[0]
            ax.set_xlim([mi, ma])
        else:
            xticks = ax.get_xticks()
            ax.set_xticks(list(range(math.floor(xticks.min()), math.ceil(xticks.max()) + 1)))
    title = 'Histogram of #tokens per sequence'
    plt.suptitle(title)
    plt.suptitle('Tokenization length distribution across datasets')
    fig.supxlabel('#token')
    fig.supylabel('density')
    if save:
        output_dir = os.path.join(PATH_BASE, DIR_PROJ, 'chore', 'plot')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{title}, {now(for_path=True)}.png'), dpi=300)
    else:
        plt.show()


plot_utcd_n_toks(extract_utcd_meta(), save=True)


if __name__ == '__main__':
    from icecream import ic

    fl_nm = 'config.json'
    ic(config)
    with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
