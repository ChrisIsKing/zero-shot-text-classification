from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

class BinaryBertCrossEncoder(CrossEncoder):
    def fit(self, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader,
    epochs: int = 1, loss_fct=None, 
    activation_fct = nn.Identity(),
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
    show_progress_bar: bool = True):

        train_dataloader.collate_fn = self.smart_batching_collate

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

        best_model = {'best_score': -9999999, 'epoch': 0, 'path' : None}


        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
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

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0
                val_steps = 0
            
                for features, labels in tqdm(val_dataloader, desc="Validation", smoothing=0.05, disable=not show_progress_bar):
                    with torch.no_grad():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        val_loss += loss_fct(logits, labels).item()
                        val_steps += 1
                
                val_loss /= val_steps
                if val_loss < best_model['best_score']:
                    best_model['best_score'] = val_loss
                    best_model['epoch'] = epoch
                    if save_best_model:
                        best_model['path'] = output_path
                        self.save(output_path)

                
       


