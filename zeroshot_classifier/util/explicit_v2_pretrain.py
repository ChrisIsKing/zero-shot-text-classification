import math
import datetime
from os.path import join as os_join

from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, TrainerCallback
import datasets

from stefutil import *


logger = get_logger('Explicit Trainer')


class MyTrainer(Trainer):
    """
    Override `compute_loss` for getting training stats
    """
    def __init__(self, name: str = None, with_tqdm: bool = True, disable_train_metrics: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.with_tqdm = with_tqdm
        self.disable_train_metrics = disable_train_metrics
        self._replace_callback()
        self.acc = datasets.load_metric('accuracy')

        d_log = dict(with_tqdm=with_tqdm, disable_train_metrics=disable_train_metrics)
        self.logger = get_logger('Explicit Trainer')
        self.logger.info(f'Trainer initialized w/ {pl.i(d_log)}')

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
        self.add_callback(MyTrainStatsMonitorCallback(trainer=self, with_tqdm=self.with_tqdm))

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
        if model.training and not self.disable_train_metrics:
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
    def __init__(self, trainer: MyTrainer, with_tqdm: bool = True):
        self.mode = 'eval'
        self.t_strt, self.t_end = None, None

        self.trainer = trainer

        self.name = self.trainer.name
        self.logger, self.logger_fl, self.writer = None, None, None
        self.ls = None

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
        output_dir = self.trainer.args.output_dir
        path_log = os_join(output_dir, f'{mdl_type} train.log')
        self.logger_fl = get_logger(name=self.name, kind='file-write', file_path=path_log)
        self.writer = SummaryWriter(os_join(output_dir, f'tb'))
        self.ls = LogStep(
            trainer=self.trainer, prettier=self.prettier,
            logger=self.logger, file_logger=self.logger_fl, tb_writer=self.writer
        )

        conf = self.trainer.model.config.to_dict()
        train_args = self.trainer.args.to_dict()
        self.logger.info(f'Training launched on model {pl.i(mdl_type)}, {pl.fmt(conf)} '
                         f'with training args {pl.fmt(train_args)}... ')
        self.logger_fl.info(f'Training launched on model {pl.i(mdl_type)}, {pl.id(conf)} '
                            f'with training args {pl.id(train_args)}... ')
        self.t_strt = datetime.datetime.now()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.t_end = datetime.datetime.now()
        t = fmt_delta(self.t_end - self.t_strt)
        self.logger.info(f'Training completed in {pl.i(t)} ')
        self.logger_fl.info(f'Training completed in {t} ')
        self.mode = 'eval'

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            step = state.global_step
            in_train = self.trainer.model.training
            if in_train and 'src' in logs and logs['src'] == 'compute_loss':
                del logs['src']
                self.out_dict = logs
            elif in_train and all('runtime' not in k for k in logs):
                d_log = dict(step=step, epoch=state.epoch, lr=logs['learning_rate'], loss=logs['loss'])
                if not self.trainer.disable_train_metrics:
                    d_log['sp_cls_acc'] = self.out_dict['acc']
                self.ls(d_log, training=in_train, to_console=not self.trainer.with_tqdm)
            elif not in_train and 'eval_loss' in logs:
                d_log = dict(step=step, epoch=int(state.epoch), loss=logs['eval_loss'], asp_cls_acc=logs['eval_acc'])
                self.ls(d_log, training=in_train, to_console=not self.trainer.with_tqdm)
            else:
                self.logger.info(pl.i(logs))
                self.logger_fl.info(pl.nc(logs))
