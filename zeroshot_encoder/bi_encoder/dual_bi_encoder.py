import random

from sentence_transformers.readers import InputExample

from zeroshot_encoder.util import *
from zeroshot_encoder.util.load_data import get_data, encoder_cls_format, in_domain_data_path
import zeroshot_encoder.bi_encoder.jskit.encoders.bi as js_bi
# Cannot import like this cos `bi.py` already imported, could cause duplicate `config_setup` call, loading 2 models
from jskit.encoders.utils.train import train_model


MODEL_NAME = 'dual-bi-encoder'
MD_NM_OUT = 'Dual Bi-encoder'


def get_train_args() -> Dict:
    # Keep the same as in `zeroshot_encoder.baseline.bi-encoder`
    return dict(  # To override `jskit.encoders.bi` defaults
        output_dir=os.path.join(get_output_base(), DIR_PROJ, DIR_MDL, MODEL_NAME, now(sep='-')),
        train_batch_size=16,  # pe `bi-encoder.py` default
        eval_batch_size=32,
        learning_rate=2e-5,  # not specified by `bi-encoder.py`, go with default `SentenceTransformer`
        num_train_epochs=3,  # per `bi-encoder.py` default
        weight_decay=1e-2,  # not specified by `bi-encoder.py`, go with default `SentenceTransformer`
        # not specified by `bi-encoder.py`, go with default `SentenceTransformer`, which uses `transformers.AdamW`
        adam_epsilon=1e-6,
        warmup_ratio=1e-1,  # per `bi-encoder.py`
    )  # Note that `jskit::train_model` internally uses a linear warmup scheduler, as in `bi-encoder.py`


def run_train(sampling: str = 'rand'):
    logger = get_logger(f'Train {MD_NM_OUT}')
    logger.info('Training launched... ')

    d_dset = get_data(in_domain_data_path)
    dnms = [dnm for dnm in d_dset.keys() if dnm != 'all']
    # dnms = list(reversed(dnms))[:2]  # TODO: debugging
    logger.info(f'Gathering datasets: {logi(dnms)}... ')
    dset_tr = sum(
        (encoder_cls_format(
            d_dset[dnm]["train"], name=dnm, sampling=sampling, neg_sample_for_multi=True, show_warnings=False
        )
         for dnm in dnms), start=[]
    )
    # dset_vl = sum((  # looks like `jskit.encoders.bi` doesn't support eval during training
    #     encoder_cls_format(dset["test"], name=dnm, train=False) for dnm, dset in d_dset if dnm != 'all'
    # ), start=[])
    n_tr = len(dset_tr)

    def batched_map(edges: Tuple[int, int]) -> Tuple[List, List, List]:  # see zeroshot_encoder.util.load_data`
        cands_tr_, conts_tr_, lbs_tr_ = [], [], []
        for i in range(*edges):
            ie: InputExample = dset_tr[i]
            cands_tr_.append(ie.texts[0])
            conts_tr_.append(ie.texts[1])
            lbs_tr_.append(ie.label)
        return cands_tr_, conts_tr_, lbs_tr_

    n_cpu = os.cpu_count()
    if n_cpu > 1 and n_tr > 2**12:
        preprocess_batch = round(n_tr / n_cpu / 2)
        strts = list(range(0, n_tr, preprocess_batch))
        ends = strts[1:] + [n_tr]  # inclusive begin, exclusive end
        cands_tr, conts_tr, lbs_tr = [], [], []
        for cd, ct, lb in conc_map(batched_map, zip(strts, ends)):
            cands_tr.extend(cd), conts_tr.extend(ct), lbs_tr.extend(lb)
        assert len(cands_tr) == n_tr
    else:
        cands_tr, conts_tr, lbs_tr = batched_map((0, n_tr))
    # n = 10
    # for c, t, l in zip(cands_tr[:n], conts_tr[:n], lbs_tr[:n]):  # Sanity check
    #     ic(c, t, l)

    train_args = get_train_args()
    out_dir, bsz_tr, bsz_vl, lr, n_ep, decay, eps, warmup_ratio = (train_args[k] for k in (
        'output_dir', 'train_batch_size', 'eval_batch_size', 'learning_rate', 'num_train_epochs',
        'weight_decay', 'adam_epsilon', 'warmup_ratio'
    ))
    assert n_tr % 3 == 0
    n_step = math.ceil(n_tr/3 / bsz_tr) * n_ep  # As 3 candidates per text, but only 1 for training

    train_params = dict(
        train_batch_size=bsz_tr, eval_batch_size=bsz_vl, num_train_epochs=n_ep, learning_rate=lr, weight_decay=decay,
        warmup_steps=round(n_step*warmup_ratio), adam_epsilon=eps
    )  # to `str` per `configparser` API
    js_bi.set_config(training_parameters={k: str(v) for k, v in train_params.items()}, model_parameters=None)
    tkzer_cnm, model_cnm = js_bi.model.__class__.__qualname__, js_bi.tokenizer.__class__.__qualname__
    shared = get(js_bi.config, 'MODEL_PARAMETERS.shared')
    gas, sz_cand, sz_cont = (js_bi.config['TRAIN_PARAMETERS'][k] for k in (
        'gradient_accumulation_steps', 'max_candidate_length', 'max_contexts_length'
    ))
    d_model = OrderedDict([
        ('model name', model_cnm), ('tokenizer name', tkzer_cnm), ('shared weights', shared == 'True'),
        ('candidate size', sz_cand), ('context size', sz_cont)
    ])
    train_args |= dict(n_step=n_step, gradient_accumulation_steps=gas)
    logger.info(f'Starting training on model {log_dict(d_model)} with training args: {log_dict(train_args)}, '
                f'dataset size: {logi(n_tr)}... ')
    model_ = train_model(
        model_train=js_bi.model,
        tokenizer=js_bi.tokenizer,
        contexts=conts_tr,
        candidates=cands_tr,
        labels=lbs_tr,
        output_dir=out_dir
    )


if __name__ == '__main__':
    import transformers
    from icecream import ic

    seed = config('random-seed')
    js_bi.set_seed(seed)
    transformers.set_seed(seed)

    def import_check():
        from zeroshot_encoder.bi_encoder.jskit.encoders.bi import (
            config as bi_enc_config, set_seed,
            tokenizer, model
        )
        ic(config_parser2dict(bi_enc_config))
        ic(tokenizer, type(model))
    # import_check()

    run_train()
