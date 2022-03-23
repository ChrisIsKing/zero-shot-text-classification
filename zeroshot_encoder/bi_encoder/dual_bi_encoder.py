from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sentence_transformers.readers import InputExample
from tqdm import tqdm

from zeroshot_encoder.util import *
from zeroshot_encoder.util.load_data import get_data, encoder_cls_format, in_domain_data_path, out_of_domain_data_path
import zeroshot_encoder.bi_encoder.jskit.encoders.bi as js_bi
import zeroshot_encoder.bi_encoder.jskit.encoders.utils as js_util
# Cannot import like this cos `bi.py` already imported, could cause duplicate `config_setup` call, loading 2 models


MODEL_NAME = 'dual-bi-encoder'
MD_NM_OUT = 'Dual Bi-encoder'


def get_train_args() -> Dict:
    # Keep the same as in `zeroshot_encoder.baseline.bi-encoder`
    return dict(  # To override `jskit.encoders.bi` defaults
        output_dir=os.path.join(get_output_base(), DIR_PROJ, DIR_MDL, MODEL_NAME, now(for_path=True)),
        train_batch_size=16,  # pe `bi-encoder.py` default
        eval_batch_size=32,
        learning_rate=2e-5,  # not specified by `bi-encoder.py`, go with default `SentenceTransformer`
        num_train_epochs=3,  # per `bi-encoder.py` default
        weight_decay=1e-2,  # not specified by `bi-encoder.py`, go with default `SentenceTransformer`
        # not specified by `bi-encoder.py`, go with default `SentenceTransformer`, which uses `transformers.AdamW`
        adam_epsilon=1e-6,
        warmup_ratio=1e-1,  # per `bi-encoder.py`
    )  # Note that `jskit::train_model` internally uses a linear warmup scheduler, as in `bi-encoder.py`


def ie_dset2js_dset(dset: List[InputExample]) -> Tuple[List, List, List]:
    """
    Convert the dataset format, from `sentence_transformers::InputExample` to the input format as in jskit training
    :return:
    """
    n_tr = len(dset)

    def batched_map(edges: Tuple[int, int]) -> Tuple[List, List, List]:  # see zeroshot_encoder.util.load_data`
        cands_tr_, conts_tr_, lbs_tr_ = [], [], []
        for i in range(*edges):
            ie: InputExample = dset[i]
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
    return cands_tr, conts_tr, lbs_tr


def run_train(sampling: str = 'rand'):
    logger = get_logger(f'Train {MD_NM_OUT}')
    logger.info('Training launched... ')

    d_dset = get_data(in_domain_data_path)
    dnms = [dnm for dnm in d_dset.keys() if dnm != 'all']
    logger.info(f'Gathering datasets: {logi(dnms)}... ')
    dset_tr = sum(
        (encoder_cls_format(
            d_dset[dnm]['train'], name=dnm, sampling=sampling, neg_sample_for_multi=True, show_warnings=False
        )
         for dnm in dnms), start=[]
    )
    # dset_vl = sum((  # looks like `jskit.encoders.bi` doesn't support eval during training
    #     encoder_cls_format(dset["test"], name=dnm, train=False) for dnm, dset in d_dset if dnm != 'all'
    # ), start=[])
    n_tr = len(dset_tr)
    cands_tr, conts_tr, lbs_tr = ie_dset2js_dset(dset_tr)

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
    not_shared_str = ''  # for not shared weights, see [ongoing-issue](https://github.com/Jaseci-Labs/jaseci/issues/150)
    js_bi.set_config(
        training_parameters={k: str(v) for k, v in train_params.items()},
        model_parameters=dict(shared=not_shared_str)
    )
    tkzer_cnm, model_cnm = js_bi.model.__class__.__qualname__, js_bi.tokenizer.__class__.__qualname__
    shared = get(js_bi.config, 'MODEL_PARAMETERS.shared')
    gas, sz_cand, sz_cont = (js_bi.config['TRAIN_PARAMETERS'][k] for k in (
        'gradient_accumulation_steps', 'max_candidate_length', 'max_contexts_length'
    ))
    d_model = OrderedDict([
        ('model name', model_cnm), ('tokenizer name', tkzer_cnm), ('shared weights', shared != not_shared_str),
        ('candidate size', sz_cand), ('context size', sz_cont)
    ])
    train_args |= dict(n_step=n_step, gradient_accumulation_steps=gas)
    logger.info(f'Starting training on model {log_dict(d_model)} with training args: {log_dict(train_args)}, '
                f'dataset size: {logi(n_tr)}... ')
    model_ = js_util.train.train_model(
        model_train=js_bi.model,
        tokenizer=js_bi.tokenizer,
        contexts=conts_tr,
        candidates=cands_tr,
        labels=lbs_tr,
        output_dir=out_dir
    )


def load_model() -> Tuple[BertTokenizer, js_util.models.BiEncoder]:
    path = os.path.join(get_output_base(), DIR_PROJ, DIR_MDL, MODEL_NAME, '2022-03-21_15-46-17')
    js_bi.load_model(path)
    return js_bi.tokenizer, js_bi.model


class MyEvalDataset(js_util.tokenizer.EvalDataset):
    def __init__(self, return_text=False, **kwargs):
        super().__init__(**kwargs)
        self.txt = kwargs['texts']
        self.return_text = return_text

    def __getitem__(self, index):
        itm = super().__getitem__(index)
        return (self.txt[index], itm) if self.return_text else itm


class EncoderWrapper:
    """
    For evaluation, a wrapper around jskit::BiEncoder
    """

    def __init__(self, model: js_util.models.BiEncoder, tokenizer: BertTokenizer):
        self.tokenizer, self.model = tokenizer, model
        self.max_cont_length = js_util.evaluate.max_contexts_length
        self.max_cand_length = js_util.evaluate.max_candidate_length

    def __call__(
            self, texts: List[str], embed_type: str, batch_size: int = 32, device: str = None, return_text=False
    ) -> Tuple[int, Iterable[torch.Tensor]]:
        """
        Yields batched embeddings in the order of `txts`
        """
        assert embed_type in ['context', 'candidate']

        if embed_type == "context":
            dset_args = dict(context_transform=js_util.tokenizer.SelectionJoinTransform(
                tokenizer=self.tokenizer, max_len=self.max_cont_length
            ))
        else:
            dset_args = dict(candidate_transform=js_util.tokenizer.SelectionSequentialTransform(
                tokenizer=self.tokenizer, max_len=self.max_cand_length
            ))
        dset = MyEvalDataset(texts=texts, **dset_args, mode=embed_type, return_text=return_text)

        def collate(samples):
            if return_text:
                txts, ins = zip(*samples)
                return list(txts), dset.eval_str(ins)
            else:
                return dset.eval_str(samples)
        dl = DataLoader(dset, batch_size=batch_size, collate_fn=collate, shuffle=False)
        # ic(len(dl))
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def callback():
            for inputs in dl:
                # for i in inputs:
                #     ic(i.shape)
                txt, inputs = inputs if return_text else (None, inputs)
                inputs = tuple(t.to(device) for t in inputs)
                input_ids, attention_masks = inputs
                # ic(input_ids.shape, attention_masks.shape)
                inputs = dict(get_embedding=embed_type, mode='get_embed')
                if embed_type == 'context':
                    inputs |= dict(context_input_ids=input_ids, context_input_masks=attention_masks)
                else:
                    inputs |= dict(candidate_input_ids=input_ids, candidate_input_masks=attention_masks)

                with torch.no_grad():
                    outputs = self.model(**inputs).squeeze(1)  # TODO: cos made changes to BiEncoder
                    yield (txt, outputs) if return_text else outputs
        return len(dl), callback()


def evaluate_trained(domain: str = 'in', candidate_batch_size: int = 256, context_batch_size: int = 32):
    assert domain in ['in', 'out']
    tokenizer, model = load_model()
    model.eval()
    ew = EncoderWrapper(model, tokenizer)
    d_dset = get_data(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
    dataset_names = [dnm for dnm in d_dset.keys() if dnm != 'all']

    # ic(type(tokenizer))
    # ic(type(model))
    domain_str = f'{domain} domain'
    output_dir = os.path.join(PATH_BASE, DIR_PROJ, 'evaluations', MODEL_NAME, f'{now(for_path=True)}, {domain_str}')

    model_cnm = model.__class__.__qualname__
    d_model = OrderedDict([
        ('model name', model_cnm), ('trained #epoch', 3),
        ('context limit', js_util.evaluate.max_contexts_length),
        ('candidate limit', js_util.evaluate.max_candidate_length),
    ])
    d_eval = OrderedDict([
        ('datasets', dataset_names),
        ('context max batch size', context_batch_size),
        ('candidate max batch size', candidate_batch_size)
    ])
    logger_name = f'{MD_NM_OUT} Evaluation'
    logger = get_logger(logger_name, typ='stdout')
    logger_fl = get_logger(
        f'{logger_name} file-write', typ='file-write',
        file_path=os.path.join(output_dir, f'{logger_name}, {domain_str}.log')
    )
    logger.info(f'Running evaluation {logi(domain_str)} on model {log_dict(d_model)}, with {log_dict(d_eval)}... ')
    logger_fl.info(f'Running evaluation {domain_str} on model {log_dict_nc(d_model)}, with {log_dict_nc(d_eval)}... ')

    # ood = domain == 'out'
    # dataset_names = [
    #     dnm for dnm in config('UTCD.datasets').keys()
    #     if (config(f'UTCD.datasets.{dnm}.out_of_domain') == ood)
    # ]
    # for dnm in dataset_names:
    for dnm in dataset_names:
        dset = d_dset[dnm]['test']

        _dset = sorted(dset)  # map from unique text to all possible labels; sort by text then label
        txt2lbs = {k: set(lb for txt, lb in v) for k, v in itertools.groupby(_dset, key=lambda pair: pair[0])}
        idx2lb = labels = sorted(set().union(*[v for v in txt2lbs.values()]))
        lb2idx = {lb: i for i, lb in enumerate(labels)}
        # ic(labels, len(labels))
        # ic(js_util.evaluate.get_embeddings(model, tokenizer, labels, embed_type='candidate'))
        # gen = ew(labels, embed_type='candidate')
        # t = next(gen)
        # ic(t.shape)
        # vects = torch.cat([e for e in ew(labels, embed_type='candidate')], dim=1)
        # ic(vects.shape)
            # ic(e.shape)
        # lst = [e for e in ew(labels, embed_type='candidate')]
        # ic(lst[0].shape, len(lst))
        vects_lb = torch.cat([e for e in ew(labels, embed_type='candidate', batch_size=candidate_batch_size)[1]])
        # emb_lbs = torch.stack(lst)
        # ic(emb_lbs.shape)
        lst_preds, lst_labels = [], []
        n, it = ew(list(txt2lbs.keys()), embed_type='candidate', return_text=True, batch_size=context_batch_size)
        logger.info(f'Running evaluation on dataset {logi(dnm)}, with labels {logi(labels)}, '
                    f'of {logi(len(txt2lbs))} unique texts in {logi(n)} batches... ')
        logger_fl.info(
            f'Running evaluation on dataset {dnm}, with labels {labels}, '
            f'of {len(txt2lbs)} unique texts in {n} batches... ')
        # ic(type(it))
        for txts, vects_txt in tqdm(it, total=n):
            # ic()
            logits = vects_txt @ vects_lb.t()
            # ic(logits.shape)
            preds = logits.argmax(dim=-1)

            def get_true_label(pred, txt):
                pos_lb_pool = txt2lbs[txt]
                if idx2lb[pred] in pos_lb_pool:
                    return pred
                else:  # Pick one true label arbitrarily if it doesn't match prediction
                    return next(lb2idx[lb] for lb in pos_lb_pool)
            lbs = torch.tensor(
                [get_true_label(p, txt) for p, txt in zip(preds.tolist(), txts)], dtype=torch.long, device=preds.device
            )
            lst_preds.append(preds)
            lst_labels.append(lbs)
            # ic(preds, lbs, len(lst_preds), len(lst_labels))
        preds_all, labels_all = torch.cat(lst_preds).cpu().numpy(), torch.cat(lst_labels).cpu().numpy()
        # ic(preds_all.shape, labels_all.shape)
        df = pd.DataFrame(
            classification_report(labels_all, preds_all, target_names=labels, output_dict=True)
        ).transpose()
        path = os.path.join(output_dir, f'{dnm}.csv')
        df.to_csv(path)
        logger.info(f'Evaluation on {logi(dnm)} written to CSV at {logi(path)}')
        logger_fl.info(f'Evaluation on {dnm} written to CSV at {path}')
        # exit(1)


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

    # run_train()

    # evaluate_trained(context_batch_size=256)
    evaluate_trained(domain='out')
