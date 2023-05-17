import logging
import os
import json
from typing import Dict, Tuple, Iterable, Type, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.losses import CosineSimilarityLoss
from tqdm.autonotebook import tqdm, trange

from stefutil import *


__all__ = ['BinaryBertCrossEncoder', 'BiEncoder']


class BinaryBertCrossEncoder(CrossEncoder):
    logger = get_logger('Bin BERT Train')

    def fit(
            self,
            train_dataloader: DataLoader = None,
            evaluator: SentenceEvaluator = None,
            # ========================== Begin of added ==========================
            val_dataloader: DataLoader = None,
            logger_fl: logging.Logger = None,
            best_model_metric: str = 'loss',
            # ========================== End of added ==========================
            epochs: int = 1, loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
    ):
        # ========================== Begin of added ==========================
        ca.check_mismatch('Eval Metric for Best Model', best_model_metric, ['loss', 'accuracy'])
        # ========================== End of added ==========================

        train_dataloader.collate_fn = self.smart_batching_collate
        # ========================== Begin of added ==========================
        if val_dataloader:
            val_dataloader.collate_fn = self.smart_batching_collate
        # ========================== End of added ==========================

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        # ========================== Begin of added ==========================
        curr_best_model = {'epoch': 0, 'best_loss': float('inf'), 'best_acc': -float('inf'), 'path': None}

        pretty = MlPrettier(ref=dict(step=len(train_dataloader), epoch=epochs))
        # ========================== End of added ==========================

        skip_scheduler = False
        # ========================== Begin of modified ==========================
        # for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        for epoch in range(epochs):
            epoch_str = f'Epoch {pl.i(epoch+1)}/{pl.i(epochs)}'
            epoch_str_nc = f'Epoch {epoch+1}/{epochs}'
            # ========================== End of modified ==========================
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            # ========================== Begin of modified ==========================
            desc = f'Training {epoch_str}'
            it = tqdm(train_dataloader, desc=desc, unit='ba', smoothing=0.05, disable=not show_progress_bar)
            for features, labels in it:
                # ========================== End of modified ==========================
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()
                # ========================== Begin of added ==========================
                # TODO: not sure why 2 lr vals, w/ same value
                d_log = dict(loss=loss_value.item(), lr=scheduler.get_last_lr()[0])
                it.set_postfix({k: pl.i(v) for k, v in pretty(d_log).items()})
                d_log = dict(epoch=epoch+1, step=training_steps+1, **d_log)
                logger_fl.info(pl.nc(pretty(d_log)))
                # ========================== End of added ==========================

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

            # ========================== Begin of added ==========================
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0
                val_steps = 0
                n_correct, n = 0, 0

                desc = f'Evaluating {epoch_str}'
                it = tqdm(val_dataloader, desc=desc, unit='ba', smoothing=0.05, disable=not show_progress_bar)
                for features, labels in it:
                    with torch.no_grad():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)

                        n_correct += (logits.argmax(dim=1) == labels).sum().item()
                        n += labels.numel()
                        val_loss += loss_fct(logits, labels).item()
                        val_steps += 1
                
                val_loss /= val_steps
                acc = n_correct / n
                d_log = pretty(dict(epoch=epoch+1, eval_loss=val_loss, eval_acc=acc))

                BinaryBertCrossEncoder.logger.info(pl.i(d_log))
                logger_fl.info(pl.nc(d_log))

                if best_model_metric == 'loss':
                    best_val = val_loss
                    prev_val = curr_best_model['best_loss']
                    better = best_val < prev_val
                else:  # `accuracy`
                    best_val = acc
                    prev_val = curr_best_model['best_acc']
                    better = best_val > prev_val
                if better:
                    curr_best_model['epoch'] = epoch+1
                    curr_best_model['best_loss' if best_model_metric == 'loss' else 'best_acc'] = best_val
                    if save_best_model:
                        curr_best_model['path'] = output_path
                        self.save(output_path)
                        BinaryBertCrossEncoder.logger.info(f'Best model found at {epoch_str} w/ '
                                                           f'{pl.i(best_model_metric)}={pl.i(best_val)} ')
                        logger_fl.info(f'Best model found at {epoch_str_nc} w/ {best_model_metric}={best_val} ')
            # ========================== End of added ==========================

        # ========================== Begin of modified ==========================
        # No evaluator, but output path: save final model version
        if val_dataloader is None and output_path is not None:
            self.save(output_path)
        # ========================== End of modified ==========================


class BiEncoder(SentenceTransformer):
    logger = get_logger('Bi-Encoder Train')

    def fit(
            self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]] = None,
            # ========================== Begin of added ==========================
            val_dataloader: DataLoader = None,
            logger_fl: logging.Logger = None,
            best_model_metric: str = 'loss',
            # ========================== End of added ==========================
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
    ):
        # ========================== Begin of added ==========================
        ca.check_mismatch('Eval Metric for Best Model', best_model_metric, ['loss', 'accuracy'])
        # ========================== End of added ==========================

        ##Add info to model card
        #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": "validation_loss", "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate
        # ========================== Begin of added ==========================
        if val_dataloader:
            val_dataloader.collate_fn = self.smart_batching_collate
        # ========================== End of added ==========================

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # ========================== Begin of added ==========================
        curr_best_model = {'epoch': 0, 'best_loss': float('inf'), 'best_acc': -float('inf'), 'path': None}

        pretty = MlPrettier(ref=dict(step=steps_per_epoch, epoch=epochs))
        # ========================== End of added ==========================

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        # ========================== Begin of added ==========================
        # for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        for epoch in range(epochs):
            epoch_str = f'Epoch {pl.i(epoch+1)}/{pl.i(epochs)}'
            epoch_str_nc = f'Epoch {epoch+1}/{epochs}'
            # ========================== End of added ==========================
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            # ========================== Begin of modified ==========================
            desc = f'Training {epoch_str}'
            it = trange(steps_per_epoch, desc=desc, unit='ba', smoothing=0.05, disable=not show_progress_bar)
            for _ in it:
                # ========================== End of modified ==========================
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()
                    # ========================== Begin of added ==========================
                    d_log = dict(loss=loss_value.item(), lr=scheduler.get_last_lr()[0])
                    it.set_postfix({k: pl.i(v) for k, v in pretty(d_log).items()})
                    d_log = dict(epoch=epoch+1, step=training_steps+1, **d_log)
                    # for k, v in d_log.items():
                    #     mic(k, v, type(k), type(v))
                    logger_fl.info(pl.nc(pretty(d_log)))
                    # ========================== End of added ==========================

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

            # ========================== Begin of added ==========================
            if val_dataloader is not None:
                self.eval()
                val_loss = 0
                val_steps = 0
                n_correct, n = 0, 0

                assert len(loss_models) == 1  # sanity check
                loss_model = loss_models[0]
                assert isinstance(loss_model, CosineSimilarityLoss)

                desc = f'Evaluating {epoch_str}'
                it = tqdm(val_dataloader, desc=desc, unit='ba', smoothing=0.05, disable=not show_progress_bar)
                for features, labels in it:
                    with torch.no_grad():
                        # See `CosineSimilarityLoss.forward`
                        embeddings = [loss_model.model(f)['sentence_embedding'] for f in features]
                        output = loss_model.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
                        loss_value = loss_model.loss_fct(output, labels.view(-1))

                        pred = (output > 0.5).long()
                        n_correct += (pred == labels).sum().item()
                        n += labels.numel()
                        val_loss += loss_value.item()
                        val_steps += 1

                val_loss /= val_steps
                acc = n_correct / n
                d_log = pretty(dict(epoch=epoch+1, eval_loss=val_loss, eval_acc=acc))

                BiEncoder.logger.info(pl.i(d_log))
                logger_fl.info(pl.nc(d_log))

                if best_model_metric == 'loss':
                    best_val = val_loss
                    prev_val = curr_best_model['best_loss']
                    better = best_val < prev_val
                else:  # `accuracy`
                    best_val = acc
                    prev_val = curr_best_model['best_acc']
                    better = best_val > prev_val
                if better:
                    curr_best_model['epoch'] = epoch+1
                    curr_best_model['best_loss' if best_model_metric == 'loss' else 'best_acc'] = best_val
                    if save_best_model:
                        curr_best_model['path'] = output_path
                        self.save(output_path)
                        BiEncoder.logger.info(f'Best model found at {epoch_str} w/ '
                                              f'{pl.i(best_model_metric)}={pl.i(best_val)} ')
                        logger_fl.info(f'Best model found at {epoch_str_nc} w/ {best_model_metric}={best_val} ')
            # ========================== End of added ==========================

        # ========================== Begin of modified ==========================
        # No evaluator, but output path: save final model version
        if val_dataloader is None and output_path is not None:
            self.save(output_path)
        # ========================== End of modified ==========================

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

