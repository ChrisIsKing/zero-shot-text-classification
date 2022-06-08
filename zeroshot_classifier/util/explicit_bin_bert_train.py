import os
import math
import datetime
from os.path import join as os_join

from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, TrainerCallback
import datasets

from stefutil import *
from zeroshot_classifier.util.util import *


class MyTrainer(Trainer):
    """
    Override `compute_loss` for getting training stats
    """
    def __init__(self, name: str = None, with_tqdm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.with_tqdm = with_tqdm
        self._replace_callback()
        self.acc = datasets.load_metric('accuracy')

    def _replace_callback(self):
        callbacks = self.callback_handler.callbacks
        # Trainer adds a `PrinterCallback` or a `ProgressCallback`, replace all that with my own,
        # see `MyProgressCallback`
        rmv = [
            "<class 'transformers.trainer_callback.ProgressCallback'>",
            "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]
        self.callback_handler.callbacks = [c for c in callbacks if str(c.__class__) not in rmv]
        if self.with_tqdm:
            self.add_callback(MyProgressCallback())
        self.add_callback(MyTrainStatsMonitorCallback(trainer=self))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Begin of added ==========================
        if model.training:
            labels_, logits = inputs['labels'].detach(), outputs.logits.detach()
            acc = self.acc.compute(predictions=logits.argmax(dim=-1), references=labels_)['accuracy']
            self.log(dict(src='compute_loss', acc=acc))
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class MyTrainStatsMonitorCallback(TrainerCallback):
    """
    Supports colored terminal output, logging file write, data sent to tensorboard for plotting

    Evaluation during training **not supported**
    """
    def __init__(self, trainer: MyTrainer):
        self.mode = 'eval'
        self.t_strt, self.t_end = None, None

        self.trainer = trainer
        paths_ = self.trainer.args.output_dir.split(os.sep)
        path_proj = paths_[paths_.index(u.proj_dir):]
        # Keep the logging & plotting inside project directory, not potentially in `scratch`
        self.output_dir = os.path.join(u.base_path, *path_proj)
        os.makedirs(self.output_dir, exist_ok=True)

        self.name = self.trainer.name
        self.logger, self.logger_fl, self.writer = None, None, None

        args = trainer.args
        n_ep = args.num_train_epochs
        bsz = args.per_device_train_batch_size * args.gradient_accumulation_steps
        n_data = len(trainer.train_dataset)
        n_step = max(math.ceil(n_data / bsz), 1) * n_ep
        self.prettier = MlPrettier(ref=dict(step=n_step, epoch=n_ep))
        self.out_dict = None

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        self.mode = 'train'

        self.logger = get_logger(self.name)
        mdl_type = self.trainer.model.__class__.__qualname__
        path_log = os_join(self.output_dir, f'{mdl_type} train.log')
        self.logger_fl = get_logger(name=self.name, typ='file-write', file_path=path_log)
        self.writer = SummaryWriter(os_join(self.output_dir, f'tb'))

        conf = self.trainer.model.config.to_dict()
        train_args = self.trainer.args.to_dict()
        self.logger.info(f'Training launched on model {logi(mdl_type)}, {log_dict_pg(conf)} '
                         f'with training args {log_dict_pg(train_args)}... ')
        self.logger_fl.info(f'Training launched on model {logi(mdl_type)}, {log_dict_id(conf)} '
                            f'with training args {log_dict_id(train_args)}... ')
        self.t_strt = datetime.datetime.now()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.t_end = datetime.datetime.now()
        t = fmt_delta(self.t_end - self.t_strt)
        self.logger.info(f'Training completed in {logi(t)} ')
        self.logger_fl.info(f'Training completed in {t} ')
        self.mode = 'eval'

    def _log(self, d, to_console: bool = True):
        d = self.prettier(d)
        if to_console:
            self.logger.info(log_dict(d))
        self.logger_fl.info(log_dict_nc(d))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            assert isinstance(logs, dict)  # sanity check
            training = self.trainer.model.training
            if training and 'src' in logs and logs['src'] == 'compute_loss':
                del logs['src']
                self.out_dict = logs
            elif training and all('runtime' not in k for k in logs):
                # Heuristics on the training step updates, see `Trainer._maybe_log_save_evaluate`
                step = state.global_step
                d_log = dict(step=step)
                assert logs['epoch'] == round(state.epoch, 2)
                d_log['epoch'] = state.epoch  # The one originally is rounded, see `Trainer.log`
                d_log['learning_rate'] = logs['learning_rate']
                # Trainer internal uses `loss`, instead of `train_loss`
                d_log['train_loss'] = loss = logs.pop('loss', None)
                assert loss is not None
                lr = d_log['learning_rate']
                d_log['train_asp_cls_acc'] = acc = self.out_dict['acc']
                self.writer.add_scalar('Train/loss', loss, step)
                self.writer.add_scalar('Train/learning_rate', lr, step)
                self.writer.add_scalar('Train/asp_cls_acc', acc, step)
                self._log(d_log, to_console=not self.trainer.with_tqdm)
            elif not training and 'eval_loss' in logs:
                loss, acc = logs['eval_loss'], logs['eval_acc']
                n_ep = state.epoch  # definitely an int
                step = state.global_step
                d_log = dict(step=step, epoch=int(n_ep), eval_loss=loss, eval_asp_cls_acc=acc)
                self.writer.add_scalar('Eval/loss', loss, step)
                self.writer.add_scalar('Eval/asp_cls_acc', acc, step)
                self._log(d_log, to_console=True)
            else:
                self.logger.info(log_dict(logs))
                self.logger_fl.info(log_dict_nc(logs))
