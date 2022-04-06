import gdown
from zipfile import ZipFile
from datasets import Value, Features, ClassLabel, Sequence, Dataset, DatasetDict

from zeroshot_encoder.util import *


def get_output_base():
    # For remote machines, save heavy-duty data somewhere else to save `/home` disk space
    hnm = get_hostname()
    if 'clarity' in hnm:  # Clarity lab
        return '/data'
    elif 'arc-ts' in hnm:  # Great Lakes; `profmars0` picked arbitrarily among [`profmars0`, `profmars1`]
        # Per https://arc.umich.edu/greatlakes/user-guide/
        return os.path.join('/scratch', 'profmars_root', 'profmars0', 'stefanhg')
    else:
        return PATH_BASE


def get_utcd_from_gdrive(domain: str = 'in'):
    assert domain in ['in', 'out']
    path = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET, 'UTCD')
    os.makedirs(path, exist_ok=True)
    if domain == 'in':
        url = 'https://drive.google.com/uc?id=1V7IzdZ9HQbFUQz9NzBDjmqYBdPd9Yfe3'
        fnm = os.path.join(path, 'in-domain')
    else:
        url = 'https://drive.google.com/uc?id=1nd32_UrFbgoCgH4bDtFFD_YFZhzcts3x'
        fnm = os.path.join(path, 'out-of-domain')
    fnm = f'{fnm}.zip'
    gdown.download(url=url, output=fnm, quiet=False)
    with ZipFile(fnm, 'r') as zip_:
        zip_.extractall(path)
        zip_.close()


def process_utcd_dataset(domain: str = 'in', join=False):
    """
    :param domain: One of [`in`, `out`]
        If 'in', process all the in-domain datasets; otherwise, process all the out-of-domain datasets
    :param join: If true, all datasets are joined to a single dataset

    .. note::
        1. The original dataset format is dictionary mapping text to list of label
        2. the datasets are processed to a multi-label format always

    Save processed datasets to disk
    """
    logger = get_logger('Process UTCD')
    assert domain in ['in', 'out'], f'Invalid domain: expect one of {logi(["in", "out"])}, got {logi(domain)}'
    output_dir = 'UTCD-in' if domain == 'in' else 'UTCD-out'
    ext = config('UTCD.dataset_ext')
    path_dsets = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
    path_out = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed')
    logger.info(f'Processing UTCD datasets with {log_dict(dict(domain=domain, join=join))}... ')

    def path2dsets(dnm: str, d_dset: Dict) -> Union[DatasetDict, Dict[str, pd.DataFrame]]:
        logger.info(f'Processing dataset {logi(dnm)}... ')
        path = d_dset['path']
        path = os.path.join(path_dsets, f'{path}.{ext}')
        with open(path) as f:
            dsets_: Dict = json.load(f)

        def json2dset(split: str, dset: Dict[str, List[str]]) -> Union[Dataset, pd.DataFrame]:
            assert split in ['train', 'test']
            if join:  # will convert to global integers later, see below
                return pd.DataFrame([dict(text=txt, labels=lbs) for txt, lbs in dset.items()])
            else:  # TODO: didn't test
                lbs_: List[str] = config(f'UTCD.datasets.{dnm}.splits.{split}.labels')
                # Map to **local** integer labels; index is label per `lbs_` ordering, same with `datasets.ClassLabel`
                lb2id = {lb: i for i, lb in enumerate(lbs_)}
                # if not multi-label, `Sequence` of single element
                df = pd.DataFrame([dict(text=txt, labels=[lb2id[lb] for lb in lbs]) for txt, lbs in dset.items()])
                length = -1 if config(f'UTCD.datasets.{dnm}.splits.{split}.multi_label') else 1
                lbs = Sequence(feature=ClassLabel(names=lbs_), length=length)
                feats = Features(text=Value(dtype='string'), labels=lbs)
                return Dataset.from_pandas(df, features=feats)
        return DatasetDict(
            {key: json2dset(key, dset) for key, dset in dsets_.items() if key not in ['labels', 'aspect']}
        )
    d_dsets = {
        dnm: path2dsets(dnm, d) for dnm, d in config('UTCD.datasets').items() if d['domain'] == domain
    }
    if join:
        dnm2id = config('UTCD.dataset_name2id')
        # Global label across all datasets, all splits
        # Needed for inversely mapping to local label regardless of joined split, e.g. train/test,
        #   in case some label only in certain split
        lbs_global = [
            config(f'UTCD.datasets.{dnm}.splits.{split}.labels')
            for dnm in d_dsets.keys() for split in ['train', 'test']
        ]
        lbs_global = sorted(set().union(*lbs_global))
        lb2id_global = {lb: i for i, lb in enumerate(lbs_global)}
        # cos definitely multi-label
        lbs_global = Sequence(feature=ClassLabel(names=lbs_global), length=-1)

        def map_labels(lbs: List[str]) -> List[int]:
            return [lb2id_global[lb] for lb in lbs]

        def prep_single(dnm: str, df_: pd.DataFrame) -> pd.DataFrame:
            df_['dataset_id'] = [dnm2id[dnm]] * len(df_)  # Add dataset source information to each row
            df_.labels = df_.labels.apply(map_labels)
            return df_

        def dfs2dset(dfs: Iterable[pd.DataFrame]) -> Dataset:
            df = pd.concat(dfs)
            # The string labels **may overlap** across the datasets
            # Keep internal feature label ordering same as dataset id
            lbs_dset = sorted(dnm2id, key=dnm2id.get)
            features = Features(text=Value(dtype='string'), labels=lbs_global, dataset_id=ClassLabel(names=lbs_dset))
            return Dataset.from_pandas(df, features=features)
        tr = dfs2dset([prep_single(dnm, dsets['train']) for dnm, dsets in d_dsets.items()])
        vl = dfs2dset([prep_single(dnm, dsets['test']) for dnm, dsets in d_dsets.items()])
        dsets = DatasetDict(train=tr, test=vl)
        dsets.save_to_disk(os.path.join(path_out, output_dir))
    else:
        for dnm, dsets in d_dsets.items():
            dsets.save_to_disk(os.path.join(path_out, dnm))
    logger.info(f'Dataset(s) saved to {logi(path_out)}')


def map_ag_news():
    dnm = 'ag_news'
    d_dset = config(f'UTCD.datasets.{dnm}')
    ext = config('UTCD.dataset_ext')
    path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
    path = d_dset['path']
    path = os.path.join(path_dset, f'{path}.{ext}')
    with open(path) as f:
        dsets: Dict = json.load(f)
    d_lb2desc = config(f'baselines.gpt2-nvidia.label-descriptors.{dnm}')
    for split, dset in dsets.items():
        dsets[split] = [[txt, d_lb2desc[lb]] for txt, lb in dset]
    with open(os.path.join(path_dset, f'{dnm}.json'), 'w') as f:
        json.dump(dsets, f, indent=4)


def get_utcd_info() -> pd.DataFrame:
    """
    Metadata about each dataset in UTCD
    """
    k_avg_tok = [f'{mode}-{text_type}_avg_tokens' for text_type in ['txt', 'lb'] for mode in ['re', 'bert', 'gpt2']]
    infos = [
        dict(dataset_name=dnm, aspect=d_dset['aspect'], domain=d_dset['domain'])
        | {f'{split}-{k}': v for split, d_info in d_dset['splits'].items() for k, v in d_info.items()}
        | {k: d_dset[k] for k in k_avg_tok}
        for dnm, d_dset in config('UTCD.datasets').items()
    ]
    return pd.DataFrame(infos)


if __name__ == '__main__':
    from icecream import ic

    from datasets import load_from_disk

    def sanity_check(dsets_nm):
        path = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', dsets_nm)
        ic(path)
        dset = load_from_disk(path)
        te, vl = dset['train'], dset['test']
        ic(len(te), len(vl))
        lbs = vl.features['labels'].feature
        ic(lbs)
        ic(vl[60])
        ic(lbs.int2str(154))
    # sanity_check('UTCD-in')

    def get_utcd_in():
        process_utcd_dataset(domain='in', join=True)
        sanity_check('UTCD-in')
    # get_utcd_in()

    def get_utcd_out():
        process_utcd_dataset(domain='out', join=True)
        sanity_check('UTCD-out')
    get_utcd_out()

    # process_utcd_dataset(in_domain=True, join=False)
    # process_utcd_dataset(in_domain=False, join=False)

    def sanity_check_ln_eurlex():
        path = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', 'multi_eurlex')
        ic(path)
        dset = load_from_disk(path)
        ic(dset, len(dset))
    # sanity_check_ln_eurlex()
    # ic(lst2uniq_ids([5, 6, 7, 6, 5, 1]))

    def output_utcd_info():
        df = get_utcd_info()
        ic(df)
        df.to_csv(os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET, 'utcd-info.csv'), float_format='%.3f')
    # output_utcd_info()

    def fix_amazon_polarity():
        """
        One test sample has 2 labels, remove it
        """
        from tqdm import tqdm
        # why doesn't pass equality????
        # wicked_txt = "This tool is absolutely fabulous for doing the few things you will need it for, but and this is " \
        #              "a BIG but (pun alert!) the replacement blades are VERY expensive--no joke, check the prices. " \
        #              "The profile sanding attachment doesn\'t work worth a darn. Because of the rotary-vibratory " \
        #              "motion of tool, the ends of the little piece of sandpaper do all the work and the center does " \
        #              "nothing. "
        wicked_lb = {'positive', 'negative'}
        path = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET, 'UTCD', 'out-of-domain', 'amazon_polarity.json')
        with open(path, 'r') as f:
            dset = json.load(f)
        wicked_txts = []
        for k, v in tqdm(dset['test'].items()):
            if len(v) > 1:
                assert set(v) == wicked_lb
                wicked_txts.append(k)
        assert len(wicked_txts) == 1
        wicked_txt = wicked_txts[0]
        ic(wicked_txt)
        # assert wicked_txt in dset['test'] and wicked_lb == set(dset['test'][wicked_txt])
        dset['test'][wicked_txt] = ['positive']
        with open(path, 'w') as f:
            json.dump(dset, f)
    # fix_amazon_polarity()
