import os
from os.path import join as os_join
from typing import Union

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
from zeroshot_classifier.preprocess import get_dataset
from zeroshot_classifier.models.gpt3 import PromptMap


HF_MODEL_NAME = 'EleutherAI/gpt-neo-2.7B'

logger = get_logger('GPT-NEO')


def evaluate(
        model_name: str = HF_MODEL_NAME, domain: str = 'in', batch_size: int = 16, dataset_name: str = 'all',
        subsample: Union[bool, int] = False, subsample_seed: int = 77, max_tokens: int = 32
):
    """
    :param model_name: Name of the GPT-Neo model
    :param domain: Dataset domain
    :param batch_size: Batch size in a generation forward pass
    :param dataset_name: Name of the dataset to evaluate
    :param subsample: Whether to subsample the dataset. If an int, the number of samples to subsample.
    :param subsample_seed: Seed for random subsampling
    :param max_tokens: # token reserved for answer
    """
    ca(dataset_domain=domain)
    tokenizer = GPT2Tokenizer.from_pretrained(HF_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(HF_MODEL_NAME)
    conf = model.config
    # conf.pad_token_id = conf.eos_token_id  # for generation
    conf.max_length = 2048  # As long as the model supports
    # from transformers import GPT2LMHeadModel
    # model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    # mic(type(model))

    model.eval()
    mic(model.device)
    import sys
    mic(fmt_sizeof(sys.getsizeof(model)))
    mic(get_model_num_trainable_parameter(model))
    if torch.cuda.is_available():
        model = model.to('cuda')

    split = 'test'
    _model_str = model_name.split('/')[-1]
    output_dir_nm = f'{now(for_path=True)}_Zeroshot-GPT-NEO-{_model_str}'
    output_path = os_join(u.eval_path, output_dir_nm, domain2eval_dir_nm(domain))
    os.makedirs(output_path, exist_ok=True)

    if dataset_name == 'all' and subsample:
        raise NotImplementedError('Subsampling intended for single dataset')
    dataset_names = utcd_util.get_eval_dataset_names(domain=domain, dataset_name=dataset_name)

    log_fnm = f'{now(for_path=True)}_GPT-NEO_{_model_str}_{domain}_{dataset_name}_Eval'
    logger_fl = get_logger('GPT3 Eval', kind='file-write', file_path=os_join(output_path, f'{log_fnm}.log'))
    d_log = dict(
        model=model_name, domain=domain, dataset_names=dataset_names, batch_size=batch_size, output_path=output_path
    )
    logger.info(f'Evaluating GPT-NEO model w/ {pl.i(d_log)}... ')
    logger_fl.info(f'Evaluating GPT-NEO model w/ {d_log}... ')

    for dnm in dataset_names:
        if subsample:
            n_tgt = subsample if isinstance(subsample, int) else 5000
            dset = utcd_util.subsample_dataset(dataset_name=dnm, split='test', n_tgt=n_tgt, seed=subsample_seed)
        else:
            dset = get_dataset(dnm, splits='test')['test']

        pm = PromptMap(dataset_name=dnm, logger_fl=logger_fl)
        # Add prompt to each text example
        dset = dset.map(lambda examples: dict(text=[pm(t) for t in examples['text']]), batched=True)
        # mic(dset, dset[0])
        # exit(1)

        map_args = dict(truncation=True, max_length=conf.max_length - max_tokens)
        dset = dset.map(lambda examples: tokenizer(examples['text'], **map_args), batched=True)

        # d_info = sconfig(f'UTCD.datasets.{dnm_}.splits.{split}')
        # lb2id = defaultdict(lambda: -1)  # If generated invalid descriptive label, will return -1
        # labels = d_info['labels']
        # # predictions and label descriptions all to lower case to be more lenient
        # lb2id.update({lb.lower(): i for i, lb in enumerate(labels)})

        n_dset = len(dset)  # See gpt2 eval, to batches of the same input id lengths
        trues, preds = np.empty(n_dset, dtype=int), np.empty(n_dset, dtype=int)
        len_ids = np.array([len(ids) for ids in dset[:]['input_ids']])
        uniq_lens = np.unique(len_ids)
        ln2idxs = [np.where(len_ids == ln)[0] for ln in uniq_lens]
        idxs_batches = sum(
            (np.split(idxs, range(batch_size, idxs.size, batch_size)) if idxs.size > batch_size else [idxs]
             for idxs in ln2idxs),
            start=[]
        )

        n_computed = 0
        it = tqdm(idxs_batches, desc=f'Evaluating {pl.i(dnm)}', unit='ba')
        for step, idxs in enumerate(it):
            idxs = [int(idx) for idx in idxs]
            inputs = {
                k: torch.tensor(v, device='cuda') for k, v in dset[idxs].items()
                if k not in ['text', 'labels']
            }
            outputs = model.generate(**inputs)
            outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            n_computed += len(idxs)
            # mic(outputs_str)
            # exit(1)

            def eval_single(generated: str = None, idx: int = None):
                idxs_boa = get_substr_indices(generated, s_sub=tokenizer.eos_token)
                idx_answ_start = len(dset[idx]['text'])
                if len(idxs_boa):
                    answer = generated[:idxs_boa[-1]]
                else:  # Did not finish, keep forward for a bit longer
                    answer = generated[:idx_answ_start + max_tokens*2]
                mic(generated, answer)
            [eval_single(g, i) for g, i in zip(outputs_str, idxs)]
            exit(1)

            def set_pred_n_true(generated: str, i_sample: int) -> Tuple[int, int]:
                idxs_boa = get_substr_indices(generated, s_sub=tokenizer.boa_token)
                # there will be at least one index, as in prompt
                if not len(idxs_boa) >= 1:
                    ids = dset[i_sample]['input_ids']
                    txt = tokenizer.decode(ids)
                    mic(generated, idxs_boa, txt)
                assert len(idxs_boa) >= 1
                # **try to be as lenient**: try to extract the text part if possible
                answer_with_eos = generated[idxs_boa[-1] + len(tokenizer.boa_token):]
                if len(idxs_boa) > 1:
                    logger.warning(f'{pl.i(model_cnm)} generated {pl.i(len(idxs_boa))} boa_token '
                                   f'instead of {pl.i(1)} with [{pl.i(answer_with_eos)}]')
                    logger_fl.warning(f'{model_cnm} generated {len(idxs_boa)} boa_token '
                                      f'instead of {1} with [{answer_with_eos}]')
                assert len(idxs_boa) == 1
                idxs_eos = get_substr_indices(answer_with_eos, s_sub=tokenizer.eos_token)
                # GPT2 would generate multiple `eos_token` for the samples in the batch that terminates early
                if len(idxs_eos) == 0:  # Still, **try to be as lenient**
                    logger.warning(f'{pl.i(model_cnm)} didn\'t finish generating answer '
                                   f'with [{pl.i(answer_with_eos)}]')
                    logger_fl.warning(f'{model_cnm} didn\'t finish generating answer with [{answer_with_eos}]')
                    answer = answer_with_eos
                else:
                    answer = answer_with_eos[:idxs_eos[0]]  # until the 1st eos
                # answer = answer.lower()
                idxs_sep = get_substr_indices(answer, s_sub=tokenizer.ques_sep_token)
                if len(idxs_sep) > 0:
                    answers = [answer[:idxs_sep[0]]]
                    for i, idx in enumerate(idxs_sep[:-1]):
                        answers.append(answer[idx + len(tokenizer.ques_sep_token):idxs_sep[i+1]])
                    answers.append(answer[idxs_sep[-1] + len(tokenizer.ques_sep_token):])
                else:
                    answers = [answer]
                ids_pred: List[int] = [lb2id[a.lower()] for a in answers]
                assert len(ids_pred) >= 1  # sanity check
                if embed_sim and all(i == -1 for i in ids_pred):  # all generated answer are non-label
                    logger.warning(f'Generated {pl.i(answers)}, not a valid label option ')
                    logger_fl.warning(f'Generate {answers}, not a valid label option ')
                    ids_pred = []
                    answ_embeds = encoder.encode(answers, batch_size=batch_size)
                    for v_ans in answ_embeds:
                        scores = [sbert_util.cos_sim(v_lb, v_ans).item() for v_lb in label_embeds]
                        ids_pred.append(int(np.argmax(scores)))
                ids_true: List[int] = dset[i_sample]['labels']
                matched = set(ids_pred) & set(ids_true)
                if len(matched) > 0:
                    # predicted label is one of the correct labels, pick that label so that prediction is correct
                    id_true = id_pred = next(iter(matched))
                else:
                    # prediction incorrect, pick a single label arbitrarily
                    # This renders class-level performance inaccurate; TODO?
                    id_pred, id_true = -1, ids_true[0]
                preds[i_sample], trues[i_sample] = id_pred, id_true
                return id_pred, id_true
            preds_batch, trues_batch = zip(*[
                set_pred_n_true(out, i_sample) for out, i_sample in zip(outputs_str, idxs)
            ])
            d_log: Dict[str, Any] = dict(
                progress=f'{n_computed:>{len(str(n_dset))}}/{n_dset}',
                sequence_length=len(inputs['input_ids'][0]),
                batch_size=f'{len(idxs):>{len(str(batch_size))}}/{batch_size}',
                n_acc=sum(p == t for p, t in zip(preds_batch, trues_batch))
            )
            it.set_postfix({k: pl.i(v) for k, v in d_log.items()})
            d_log.update(dict(ids_pred=list(preds_batch), ids_true=list(trues_batch)))
            logger_fl.info(pl.nc(d_log))

        def check_labels_filled(lbs):  # sanity check, every index is assigned a label
            return np.all((-1 <= lbs) & (lbs < len(labels)))
        assert check_labels_filled(trues) and check_labels_filled(preds)

        # note `-1` is not actual label, support of 0 - included for full label specification per sklearn
        # **note** cos the -1 label, the `macro avg` row is not accurate;
        # included it for getting global accuracy
        args = dict(
            labels=[-1, *range(len(labels))], target_names=['Label not in dataset', *labels],
            zero_division=0, output_dict=True  # disables warning
        )
        report = classification_report(trues, preds, **args)
        acc = f'{report["accuracy"]:.3f}'
        logger.info(f'{pl.i(dnm_)} Classification Accuracy: {pl.i(acc)}')
        logger_fl.info(f'{dnm_} Classification Accuracy: {acc}')

        df = pd.DataFrame(report).transpose()
        path = os_join(output_path, f'{dnm_}.csv')
        df.to_csv(path)
        logger.info(f'Evaluation on {pl.i(dnm_)} written to CSV at {pl.i(path)}')
        logger_fl.info(f'Evaluation on {dnm_} written to CSV at {path}')


if __name__ == '__main__':
    evaluate(domain='in', dataset_name='emotion')
