import os
import sys
import math
import logging
import datetime
from os.path import join as os_join
from time import sleep
from typing import Dict, Tuple, List, Union, Optional
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast
from transformers import TrainingArguments, TrainerCallback, Trainer
from transformers.trainer_utils import EvalLoopOutput
from transformers.file_utils import is_torch_tpu_available
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

from stefutil import *
from zeroshot_classifier.util.util import *
from zeroshot_classifier.util.training import *


class MyLoggingCallback(TrainerCallback):
    """
    Requires
        - Tuple of (custom compute_loss log, internal training log, internal validation log) for each step
            - Intended for coupled training and evaluation
        - Accuracy as a metric is passed to `Trainer` and training metric computed in `compute_loss` and logged
    """
    def __init__(
            self, parent_trainer: Trainer, do_eval=True,
            name='GPT2-NVIDIA-Train', is_ddp: Union[int, bool] = False
    ):
        """
        :param parent_trainer: The parent Trainer
        :param name: Logger name
        :param is_ddp: Flag for if distributed training is used
            So that logging step is correct, since each scrip only see 1 GPU
        """
        self.name = name
        self.out_dict = None
        self.out_dict_tr = None
        self.is_compute_loss_on_train = True
        self.k_acc = 'acc_meta'
        self.k_cls = 'cls_acc_meta'  # See `CustomTrainer`
        self.k_cls_eval = f'{self.k_cls}_eval'

        self.trainer = parent_trainer
        args, dset_tr__, dset_vl_, md_, tokzer = (
            getattr(parent_trainer, k) for k in ['args', 'train_dataset', 'eval_dataset', 'model', 'tokenizer']
        )
        self.n_eval = len(dset_vl_)
        lr, n_ep = args.learning_rate, args.num_train_epochs
        self.bsz = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.is_ddp = is_ddp
        if is_ddp:
            assert isinstance(is_ddp, int), 'When DDP enabled, is_ddp must specify #GPU'
            self.bsz = self.bsz * is_ddp
        if torch.cuda.is_available() and self.trainer.args.n_gpu > 1:
            self.bsz *= self.trainer.args.n_gpu
        seq_max_len = len(dset_tr__[0]['input_ids'])
        n_data, md_sz = len(dset_tr__), md_.config.n_positions
        self.n_step = max(math.ceil(n_data / self.bsz), 1) * n_ep  # #step/epoch at least 1
        self.train_meta = OrderedDict([
            ('#data', n_data), ('model size', md_sz),
            ('learning rate', lr), ('batch shape', (self.bsz, seq_max_len)), ('#epochs', n_ep), ('#steps', self.n_step),
        ])
        self.prettier = MlPrettier(ref=self.train_meta)
        self.called_val_init = False

        self.do_eval = do_eval

        self.save_time = now(for_path=True)
        self.logger, self.logger_fl, self.tb_writer = None, None, None
        self.ls = None

        self.log_fnm = f'{name}_{{{pl.pa(dict(n=n_data, l=md_sz, a=lr, bsz=self.bsz, n_ep=n_ep))}}}'

        self.train_begin, self.train_end = None, None
        self.t_strt, self.t_end = None, None

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        if self.trainer.is_local_process_zero():  # For distributed training; TODO: support multi machine?
            self.logger: logging.Logger = get_logger(self.name)
            output_dir = self.trainer.args.output_dir
            fl_path = os_join(output_dir, f'{self.log_fnm}.log')
            self.logger_fl = get_logger(name=self.name, kind='file-write', file_path=fl_path)
            self.tb_writer = SummaryWriter(os_join(output_dir, f'TB_{self.log_fnm}'))
            self.ls = LogStep(
                trainer=self.trainer, prettier=self.prettier,
                logger=self.logger, file_logger=self.logger_fl, tb_writer=self.tb_writer
            )

            conf = self.trainer.model.config.to_dict()
            args = self.trainer.args.to_dict()
            sleep(2)  # otherwise, logging messages missing
            self.logger.info(f'Training started on model{pl.fmt(conf)}, {pl.i(self.train_meta)} and '
                             f'training args: {pl.fmt(args)}...  ')
            self.logger_fl.info(f'Training started on model{pl.id(conf)}, {pl.nc(self.train_meta)} and '
                                f'training args: {pl.id(args)}...  ')
            sleep(2)
            self.t_strt = datetime.datetime.now()

            self.train_begin = True

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        if self.train_begin:
            self.train_begin = False
            self.train_end = True

            self.t_end = datetime.datetime.now()
            t = fmt_delta(self.t_end - self.t_strt)
            self.logger.info(f'Training completed in {pl.i(t)} ')
            self.logger_fl.info(f'Training completed in {t} ')

    def on_evaluate(self, args: TrainingArguments, state, control, **kwargs):
        if self.trainer.is_local_process_zero():  # Similarly to `on_train_begin`
            dl_vl: DataLoader
            model, dl_vl = kwargs['model'], kwargs['eval_dataloader']
            dset_vl: Dataset = dl_vl.dataset
            n_eval = len(dset_vl)
            bsz = dl_vl.batch_size
            seq_max_len = len(dset_vl[0]['input_ids'])
            md_sz = model.config.n_positions
            n_bch = max(math.ceil(n_eval / bsz), 1)
            eval_meta = OrderedDict([
                ('#data', n_eval), ('model size', md_sz), ('batch shape', (bsz, seq_max_len)), ('#batches', n_bch)
            ])
            if not self.trainer.with_tqdm:
                self.logger.info(f'Ran evaluation with {pl.i(eval_meta)}')
            self.logger_fl.info(f'Ran evaluation with {pl.nc(eval_meta)}')

    def _acc_stats2dict(self, out_dict: Dict) -> Dict:
        """
        Convert `acc_meta`, `classification_acc_meta` dict to stats for logging
        """
        stats_acc = {k: sum(d[k] for d in out_dict[self.k_acc]) for k in out_dict[self.k_acc][0].keys()}
        del out_dict[self.k_acc]
        ret = dict(ntp_acc=stats_acc['n_acc'] / stats_acc['n_total'])
        if self.k_cls in out_dict:
            stats_cls_acc = {
                k: sum(d[k] for d in out_dict[self.k_cls]) for k in out_dict[self.k_cls][0].keys()
                if k in ['n_acc', 'n_total']
            }

            del out_dict[self.k_cls]
            if stats_cls_acc['n_total'] == 0:
                cls_acc = 0
            else:
                cls_acc = (stats_cls_acc['n_acc']/stats_cls_acc['n_total'])
            ret['cls_acc'] = cls_acc
        return ret

    def on_log(self, args: TrainingArguments, state, control, logs: Dict = None, **kwargs):
        # basically only log the main process; `state.is_local_process_zero` is wrong in DDP eval
        if self.trainer.is_local_process_zero():
            step = state.global_step
            if 'src' in logs and logs['src'] == 'compute_loss':
                # For gradient_accumulation, many batches of `compute_loss` may be called,
                # before going into train logging
                # Loss here is per batch, not per gradient update, ignore
                if self.out_dict_tr is None:
                    n_ep = logs['epoch']
                    self.out_dict_tr = {'step': step, 'epoch': n_ep, self.k_acc: [logs[self.k_acc]]}
                    # Aggregate accuracy & classification accuracy counts
                    if self.trainer.compute_cls_acc:
                        self.out_dict_tr[self.k_cls] = [logs[self.k_cls]]
                else:  # Later batch in the same gradient accumulation
                    step_, n_ep = self.out_dict_tr['step'], self.out_dict_tr['epoch']
                    n_ep_ = logs['epoch']
                    assert step_ == step and n_ep_ == n_ep
                    self.out_dict_tr[self.k_acc].append(logs[self.k_acc])
                    if self.trainer.compute_cls_acc:
                        self.out_dict_tr[self.k_cls].append(logs[self.k_cls])
            elif 'loss' in logs:  # Trainer default training loss logging
                d_log = dict(epoch=state.epoch, step=step+1)  # 1-indexed
                d_log.update(dict(lr=logs['learning_rate'], loss=logs['loss']))
                if not self.trainer.disable_train_metrics:
                    d_log.update(self._acc_stats2dict(self.out_dict_tr))
                self.ls(d_log, training=True, to_console=not self.trainer.with_tqdm)
                self.out_dict_tr = None  # Reset for next global step
            elif 'eval_loss' in logs:  # Trainer eval output after eval metric computed
                n_ep = logs['epoch']
                assert n_ep.is_integer()
                d_log = dict(epoch=int(n_ep), loss=logs['eval_loss'], cls_acc=logs['eval_cls_acc'])
                self.ls(d_log, training=False, to_console=not self.trainer.with_tqdm)
            else:
                self.logger.info(pl.i(logs))
                self.logger_fl.info(pl.nc(logs))


class ColoredPrinterCallback(TrainerCallback):
    def __init__(self, name='Zero-shot GPT-2 Training'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        had_handler = False
        hd_attr_nm = 'name_for_my_logging'
        for hd in self.logger.handlers:
            if hasattr(hd, hd_attr_nm) and getattr(hd, hd_attr_nm) == name:
                had_handler = True
        if not had_handler:
            handler = logging.StreamHandler(stream=sys.stdout)  # For my own coloring
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(MyFormatter())
            setattr(handler, hd_attr_nm, name)
            self.logger.addHandler(handler)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(pl.i(logs) if isinstance(logs, dict) else logs)


def get_accs(
        inputs: Dict[str, torch.Tensor], logits: torch.Tensor, tokenizer: GPT2TokenizerFast, mode: str = 'train',
        compute_cls_acc: bool = False
) -> Dict:
    """
    :param inputs: Dictionary of a 2D batch of input tensors, with keys [`labels`, `token_type_ids`, `dataset_id`]
    :param logits: logits by ZsGPT2LMHeadModel's forward pass
    :param tokenizer: ZsGPT2Tokenizer for getting the class label
    :param mode: Determines which split the labels are from, one of [`train`, `eval`]
    :param compute_cls_acc: Whether to compute classification accuracy
    :return: NTP accuracy & sample classification accuracy metadata

    .. note: Classification accuracy based on NTP task **during training**
        **assumes** predicted token id at the same location of label id
    """
    preds = logits.argmax(dim=-1)
    labels_ = inputs['labels'].detach()
    # CLM, predicting the next token given current, so shift
    # Last prediction is not part of input label, 1st input is fed into model & not predicted
    preds, labels_ = preds[:, :-1], labels_[:, 1:]
    mask_non_pad = labels_ != PT_LOSS_PAD  # Consider only the actual tokens for accuracy
    preds_non_pad, labels_non_pad = preds[mask_non_pad], labels_[mask_non_pad]
    matches: torch.Tensor = (preds_non_pad == labels_non_pad)
    d_ret = dict(acc_meta=dict(n_acc=matches.sum().item(), n_total=preds_non_pad.numel()))

    if compute_cls_acc:
        token_type_ids, dataset_id = inputs['token_type_ids'].detach(), inputs['dataset_id'].detach()

        id_att = tokenizer.enc_spec(tokenizer.answer_type_token)
        id_answ = tokenizer.enc_spec(tokenizer.boa_token)
        id_eos = tokenizer.enc_spec(tokenizer.eos_token)
        # Also shift by 1
        lst_idxs_answ: List[List[int]] = [(row == id_att).nonzero().flatten().tolist() for row in token_type_ids[:, 1:]]

        id_sep = tokenizer.encode(tokenizer.ques_sep_token)[0]

        def get_label_ids(i_sample: int, idxs_answ: List[int]) -> List[Tuple[int, List[int]]]:
            """
            Prepare for input to `get_label_id`
            :param i_sample: Index of sample as in `input_ids`
            :param idxs_answ: Indices of the answer part
            :return: Potentially breaks down the indices of list of labels into sublists, one for each label
            """
            msk_sep: torch.Tensor = labels_[i_sample, idxs_answ] == id_sep
            if torch.any(msk_sep):
                idxs_sep = msk_sep.nonzero().flatten().tolist()
                # filters out the sep token
                idxs = [*idxs_sep, None]
                lst_idxs_answ_ = [idxs_answ[:idxs_sep[0]]]
                lst_idxs_answ_ += [idxs_answ[idx+1:idxs[i+1]] for i, idx in enumerate(idxs[:-1])]
                return [(i_sample, idxs_answ_) for idxs_answ_ in lst_idxs_answ_]
            else:
                return [(i_sample, idxs_answ)]

        def get_label_id(i_sample: int, idxs_answ: List[int]) -> Dict[str, int]:
            """
            :return: classification label predicted & expected

            .. note:: answer tokens should be present in each row/sample
            """
            assert len(idxs_answ)  # Should always exist, see `ZsGPT2Tokenizer.__call__`
            token_ids_true = labels_[i_sample, idxs_answ].tolist()  # Inputs are labels
            # Remove answer special prefix token & potentially the ending token
            if token_ids_true[0] == id_answ:
                idxs_answ, token_ids_true = idxs_answ[1:], token_ids_true[1:]
            assert len(token_ids_true)  # Labels should always be available
            if token_ids_true[-1] == id_eos:
                idxs_answ, token_ids_true = idxs_answ[:-1], token_ids_true[:-1]
                assert len(token_ids_true)

            dset_id = dataset_id[i_sample].item()
            dnm_ = sconfig('UTCD.dataset_id2name')[dset_id]
            split = 'train' if mode == 'train' else 'test'
            descs = sconfig(f'UTCD.datasets.{dnm_}.splits.{split}.labels')
            desc_true = tokenizer.decode(token_ids_true)
            assert desc_true in descs
            # By default, the predictions and labels will not agree
            d_lbs_ = dict(label_id_pred=-1, label_id_true=descs.index(desc_true))  # Local label wrt dataset
            desc_pred = tokenizer.decode(preds[i_sample, idxs_answ])
            if desc_pred in descs:
                d_lbs_['label_id_pred'] = descs.index(desc_pred)
            return d_lbs_

        args = sum([get_label_ids(i_sample, idxs_answ) for i_sample, idxs_answ in enumerate(lst_idxs_answ)], start=[])
        lst_idxs_n_lbs = [get_label_id(*a) for a in args]
        d_lbs: Dict[str, List[int]] = {k_id: [d[k_id] for d in lst_idxs_n_lbs] for k_id in lst_idxs_n_lbs[0].keys()}
        ids_pred, ids_true = d_lbs['label_id_pred'], d_lbs['label_id_true']
        n_acc = sum(p == t for p, t in zip(ids_pred, ids_true))  # prediction ids match label ids
        n_total = len(ids_true)  # note multi-label means potentially more classification denominator than batch size
        d_ret['cls_acc_meta'] = dict(n_acc=n_acc, n_total=n_total, ids_pred=ids_pred, ids_true=ids_true)
    return d_ret


class MyTrainer(Trainer):
    def __init__(
            self, tokenizer: GPT2TokenizerFast = None, custom_logging=True,
            disable_train_metrics: bool = True, compute_cls_acc: bool = False,
            is_ddp: Union[bool, int] = False, with_tqdm: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        assert 'args' in kwargs

        self.custom_logging = custom_logging
        # Calling `get_accs` during training seems to reduce GPU util
        self.disable_train_metrics = disable_train_metrics
        self.compute_cls_acc = compute_cls_acc
        self.is_ddp = is_ddp
        self.with_tqdm = with_tqdm

        self.tokenizer = tokenizer  # TODO: generalize to more tokenizers?
        self.mode = None

        self.post_init()
        # Sanity check for distributed training
        print(f'Trainer instantiated with is_local_process_zero: {pl.i(self.is_local_process_zero())}')

        self.logger = get_logger('GPT2 Trainer')
        d_log = dict(
            custom_logging=custom_logging, disable_train_metrics=disable_train_metrics,
            compute_cls_acc=compute_cls_acc, is_ddp=is_ddp, with_tqdm=with_tqdm
        )
        self.logger.info(f'Trainer initialized w/ {pl.i(d_log)}')

    def post_init(self):
        callbacks = self.callback_handler.callbacks
        self.callback_handler.callbacks = [  # Remove internal callback
            c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]

        if self.custom_logging:
            self.add_callback(MyLoggingCallback(self, do_eval=self.args.do_eval, is_ddp=self.is_ddp))
        else:
            self.add_callback(ColoredPrinterCallback())
        if self.with_tqdm:
            self.add_callback(MyProgressCallback())

    def train(self, **kwargs):
        self.mode = 'train'
        return super().train(**kwargs)

    def evaluate(self, **kwargs):
        if not self.is_in_train:
            self.mode = 'eval'
        return super().evaluate(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override `Trainer.compute_loss` for logging accuracy
            - Note that both training and validation calls `compute_loss`
                => Further logic needs to determine accuracy for which dataset

        Modified from https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4?u=stefanh
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Begin of added ==========================
        inputs: Dict[str, torch.Tensor]
        if self.custom_logging and model.training and 'labels' in inputs and (not self.disable_train_metrics):
            d_log = get_accs(
                inputs, outputs.logits.detach(), self.tokenizer, mode=self.mode, compute_cls_acc=self.compute_cls_acc
            )
            d_log['src'] = 'compute_loss'
            self.log(d_log)
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """
        Override `Trainer.prediction_step` for reducing memory footprint
        """
        # ========================== Begin of added =========================
        from transformers.file_utils import is_sagemaker_mp_enabled
        from transformers.trainer_pt_utils import nested_detach
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
        # ========================== End of added =========================

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        # ========================== Begin of added =========================
        if not self.model.training and self.compute_cls_acc:
            # Compute the labels right away,
            # instead of potentially concatenating the original evaluation matrix of shape (#eval, #model size, #vocab)
            # shape now is (#eval) cos for classification
            # During training, the eval set has the same set of labels as the training set,
            # which is the sole purpose of `mode`
            d_acc = get_accs(inputs, logits, self.tokenizer, mode='train', compute_cls_acc=self.compute_cls_acc)
            args = dict(dtype=labels.dtype, device=labels.device)  # For DDP
            # TODO: log token-level ACC too?
            return (
                loss,
                torch.tensor(d_acc['cls_acc_meta']['ids_pred'], **args),
                torch.tensor(d_acc['cls_acc_meta']['ids_true'], **args),
                inputs['dataset_id'].detach()
            )
        else:
            return loss, logits, labels, None
    # ========================== End of added =========================

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        For sending `dataset_id` to evaluate
        """
        # ========================== Begin of added =========================
        from transformers.deepspeed import deepspeed_init
        import collections
        from transformers.trainer_utils import denumpify_detensorize
        from torch.utils.data import IterableDataset
        from transformers.trainer_pt_utils import (
            find_batch_size, nested_concat, nested_numpify, nested_truncate, IterableDatasetShard
        )
        # ========================== End of added =========================
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        # ========================== Begin of added =========================
        from transformers.utils import logging
        logger = logging.get_logger(__name__)
        # ========================== End of added =========================
        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # ========================== Begin of added =========================
        dataset_ids_host = None
        # ========================== End of added =========================
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # ========================== Begin of added =========================
        all_dataset_ids = None
        # ========================== End of added =========================
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, dataset_ids = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            # ========================== Begin of added =========================
            if dataset_ids is not None:
                dataset_ids = self._pad_across_processes(dataset_ids)
                dataset_ids = self._nested_gather(dataset_ids)
                dataset_ids_host = (
                    dataset_ids if dataset_ids_host is None
                    else nested_concat(dataset_ids_host, dataset_ids, padding_index=-100)
                )
            # ========================== End of added =========================
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # ========================== Begin of added =========================
                if dataset_ids_host is not None:
                    dataset_ids = nested_numpify(dataset_ids_host)
                    all_dataset_ids = (
                        dataset_ids if all_dataset_ids is None
                        else nested_concat(all_dataset_ids, dataset_ids, padding_index=-100)
                    )
                # ========================== End of added =========================

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        # ========================== Begin of added =========================
        if dataset_ids_host is not None:
            dataset_ids = nested_numpify(dataset_ids_host)
            all_dataset_ids = (
                dataset_ids if all_dataset_ids is None
                else nested_concat(dataset_ids, all_dataset_ids, padding_index=-100)
            )
        # ========================== End of added =========================

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        # ========================== Begin of added =========================
        if all_dataset_ids is not None:
            all_dataset_ids = nested_truncate(all_dataset_ids, num_samples)
        # ========================== End of added =========================

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            # ========================== Begin of modified =========================
            if self.is_local_process_zero():
                mep = MyEvalPrediction(predictions=all_preds, label_ids=all_labels, dataset_ids=all_dataset_ids)
                metrics = self.compute_metrics(mep)
            else:
                metrics = {}
            # ========================== End of modified =========================
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

