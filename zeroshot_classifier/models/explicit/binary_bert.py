import os
import math
import logging
import datetime
from os.path import join
from os.path import join as os_join
from typing import List, Type, Dict, Optional, Union
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    BertConfig, BertModel, BertPreTrainedModel, BertTokenizer,
    get_scheduler
)
from sklearn.metrics import classification_report
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from zeroshot_classifier.util.load_data import (
    get_datasets, binary_explicit_format, in_domain_data_path, out_of_domain_data_path
)
from stefutil import *
from zeroshot_classifier.util import *


logger = logging.getLogger(__name__)
set_seed(42)


CLS_LOSS_ONLY = True  # TODO: debugging


class BertZeroShotExplicit(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.binary_cls = nn.Linear(self.config.hidden_size, 2)
        self.aspect_cls = None if CLS_LOSS_ONLY else nn.Linear(self.config.hidden_size, 3)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # calls `from_pretrained` from class `PreTrainedModel`
        obj = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return obj
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        binary_logits = self.binary_cls(pooled_output)
        # if CLS_LOSS_ONLY:
        #     with torch.no_grad():
        #         aspect_logits = self.aspect_cls(pooled_output)
        # else:
        #     aspect_logits = self.aspect_cls(pooled_output)
        aspect_logits = None if CLS_LOSS_ONLY else self.aspect_cls(pooled_output)
        
        loss = None
        
        logits = {'cls': binary_logits, 'aspect': aspect_logits}
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits, outputs.hidden_states, outputs.attentions


class ExplicitCrossEncoder:
    def __init__(self, name="bert-base-uncased", device: Union[str, torch.device] = 'cuda', max_length=None) -> None:
        self.config = BertConfig.from_pretrained(name)
        self.model = BertZeroShotExplicit(self.config)
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.device = device
        self.max_length = max_length

        self.writer = None
        self.model_meta = dict(model='BinaryBERT', mode='explicit')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        obj = cls()
        obj.model = BertZeroShotExplicit.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return obj
    
    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        aspects = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)
            aspects.append(example.aspect)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        aspects = torch.tensor(aspects, dtype=torch.long).to(self.device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized, labels, aspects

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized

    def fit(
            self,
            train_dataloader: DataLoader,
            epochs: int = 1,
            scheduler: str = 'linear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            output_path: str = None,
            max_grad_norm: float = 1,
            show_progress_bar: bool = True
    ):
        os.makedirs(output_path, exist_ok=True)
        mdl, md = self.model_meta['model'], self.model_meta['mode']
        log_fnm = f'{now(for_path=True)}, {mdl}, md={md}, #ep={epochs}'
        self.writer = SummaryWriter(os_join(output_path, f'tb - {log_fnm}'))

        train_dataloader.collate_fn = self.smart_batching_collate
        self.model.to(self.device)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        num_training_steps = int(len(train_dataloader) * epochs)

        lr_scheduler = get_scheduler(
            name=scheduler, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )

        def _get_lr() -> float:
            return lr_scheduler.get_last_lr()[0]

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            tr_loss = 0
            self.model.zero_grad()
            self.model.train()

            with tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar) as it:
                for features, labels, aspects in it:
                    model_predictions = self.model(**features, return_dict=True)

                    pooled_output = model_predictions[1]
                    loss_fct = CrossEntropyLoss()

                    # if CLS_LOSS_ONLY:
                    #     with torch.no_grad():
                    #         task_loss_value = loss_fct(pooled_output['aspect'].view(-1, 3), aspects.view(-1))
                    # else:
                    #     task_loss_value = loss_fct(pooled_output['aspect'].view(-1, 3), aspects.view(-1))
                    task_loss_value = None
                    if not CLS_LOSS_ONLY:
                        task_loss_value = loss_fct(pooled_output['aspect'].view(-1, 3), aspects.view(-1))
                    binary_loss_value = loss_fct(pooled_output['cls'].view(-1, 2), labels.view(-1))

                    cls_loss = binary_loss_value.detach().item()
                    asp_loss = None if CLS_LOSS_ONLY else task_loss_value.detach().item()
                    it.set_postfix(cls_loss=cls_loss, asp_loss=asp_loss)
                    step = training_steps + epoch * len(train_dataloader)
                    self.writer.add_scalar('Train/learning rate', _get_lr(), step)
                    self.writer.add_scalar('Train/Binary Classification Loss', cls_loss, step)
                    if not CLS_LOSS_ONLY:
                        self.writer.add_scalar('Train/Aspect Classification Loss', asp_loss, step)
                    if CLS_LOSS_ONLY:
                        loss = binary_loss_value
                    else:
                        loss = task_loss_value + binary_loss_value
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    training_steps += 1
                    tr_loss += loss.item()
            
            average_loss = tr_loss/training_steps
            print(f'Epoch: {epoch+1}\nAverage loss: {average_loss:f}\n Current Learning Rate: {lr_scheduler.get_last_lr()}')
    
        self.save(output_path)

    def predict(self, sentences: List[List[str]], batch_size: int = 32):
        
        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, shuffle=False)

        show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        show_progress_bar = False

        iterator = inp_dataloader
        
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")
        
        pred_scores = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = model_predictions[1]['cls']

                if len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        # pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
        pred_scores = torch.stack(pred_scores, dim=0).detach().cpu()

        return pred_scores

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def parse_args():
    modes = [
        'vanilla',
        'implicit',
        'implicit-on-text-encode-aspect',  # encode each of the 3 aspects as 3 special tokens, followed by text
        'implicit-on-text-encode-sep',  # encode aspects normally, but add special token between aspect and text
        'explicit'
    ]

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--output', type=str, required=True)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], required=True)
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    # set test arguments
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_test.add_argument('--batch_size', type=int, default=32)
    parser_test.add_argument('--model_path', type=str, required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    import transformers

    transformers.logging.set_verbosity_error()  # disables `longest_first` warning

    args = parse_args()
    mode = args.mode
    assert mode == 'explicit'

    if args.command == 'train':

        bsz, lr, n_ep = args.batch_size, args.learning_rate, args.epochs
        sampling = args.sampling
        dirs = args.output.split(os.sep)
        dir_nm_last = f'{now(for_path=True)}-{dirs[-1]}-{sampling}-{args.mode}'
        save_path = os_join(*dirs[:-1], dir_nm_last)
        _logger = get_logger('BinaryBERT Explicit Training')
        d_log = dict(mode=mode, sampling=sampling, batch_size=bsz, epochs=n_ep, learning_rate=lr, save_path=save_path)
        _logger.info(f'Running training on {pl.i(d_log)}.. ')

        dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

        _logger.info('Loading data & model... ')
        # n_sample = 1024 * 8  # TODO: debugging
        n_sample = None
        data = get_datasets(in_domain_data_path, n_sample=n_sample)
        # get keys from data dict
        datasets = list(data.keys())
        train = binary_explicit_format(data)

        dl = DataLoader(train, shuffle=True, batch_size=bsz)

        model = ExplicitCrossEncoder('bert-base-uncased', device=dvc)

        warmup_steps_ = math.ceil(len(dl) * n_ep * 0.1)  # 10% of train data for warm-up
        _logger.info(f'Launched training on {pl.i(len(train))} samples and {pl.i(warmup_steps_)} warmup steps... ')

        model.fit(
            train_dataloader=dl,
            epochs=n_ep,
            warmup_steps=warmup_steps_,
            optimizer_params={'lr': lr},
            output_path=save_path
        )
    if args.command == 'test':
        mode, domain, model_path, bsz = args.mode, args.domain, args.model_path, args.batch_size
        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        date = datetime.datetime.now().strftime('%m.%d.%Y')
        date = date[:-4] + date[-2:]  # 2-digit year
        out_path = join(model_path, 'eval', f'{domain_str}, {date}')
        os.makedirs(out_path, exist_ok=True)

        data = get_datasets(in_domain_data_path if domain == 'in' else out_of_domain_data_path)
        model = ExplicitCrossEncoder.from_pretrained(model_path)  # load model
        sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')
        aspect2aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')

        logger = get_logger('Binary Bert Eval')
        d_log = dict(mode=mode, domain=domain, batch_size=bsz, path=model_path)
        logger.info(f'Evaluating Binary Bert with {pl.i(d_log)} and saving to {pl.i(out_path)}... ')

        eval_loss: Dict[str, np.array] = dict()  # a sense of how badly the model makes the prediction
        dataset_names = [dnm for dnm, d_dset in sconfig('UTCD.datasets').items() if d_dset['domain'] == domain]

        for dnm in dataset_names:  # loop through all datasets
            # if 'consumer' not in dnm:
            #     continue
            dset = data[dnm]
            split = 'test'
            txts, aspect = dset[split], dset['aspect']
            d_dset = sconfig(f'UTCD.datasets.{dnm}.splits.{split}')
            label_options, multi_label = d_dset['labels'], d_dset['multi_label']
            n_options = len(label_options)
            label2id = {lbl: i for i, lbl in enumerate(label_options)}
            n_txt = sconfig(f'UTCD.datasets.{dnm}.splits.{split}.n_text')
            d_log = {'#text': n_txt, '#label': n_options}
            logger.info(f'Evaluating {pl.i(dnm)} with {pl.i(d_log)}...')
            arr_preds, arr_labels = np.empty(n_txt, dtype=int), np.empty(n_txt, dtype=int)

            txt_n_lbs2query = None
            if mode in ['vanilla', 'explicit']:
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, lb] for lb in lbs]
            elif mode == 'implicit':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, f'{lb} {aspect}'] for lb in lbs]
            elif mode == 'implicit-on-text-encode-aspect':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect2aspect_token[aspect]} {txt}', lb] for lb in lbs]
            elif mode == 'implicit-on-text-encode-sep':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect} {sep_token} {txt}', lb] for lb in lbs]

            gen = group_n(txts.items(), n=bsz)
            # loop through each test example
            for i_grp, group in enumerate(tqdm(gen, desc=dnm, unit='group', total=math.ceil(n_txt/bsz))):
                txts_, lst_labels = zip(*group)
                lst_labels: List[List[int]] = [[label2id[lb] for lb in labels] for labels in lst_labels]
                query = sum([txt_n_lbs2query(t, label_options) for t in txts_], start=[])  # (n_options x bsz, 2)
                # probability for positive class
                logits = model.predict(query, batch_size=bsz)[:, 1]
                logits = logits.reshape(-1, n_options)
                preds = logits.argmax(axis=1)
                trues = torch.empty_like(preds)
                for i, pred, labels in zip(range(bsz), preds, lst_labels):
                    # if false prediction, pick one of the correct labels arbitrarily
                    trues[i] = pred if pred in labels else labels[0]
                idx_strt = i_grp*bsz
                arr_preds[idx_strt:idx_strt+bsz], arr_labels[idx_strt:idx_strt+bsz] = preds.cpu(), trues.cpu()

            args = dict(zero_division=0, target_names=label_options, output_dict=True)  # disables warning
            report = classification_report(arr_labels, arr_preds, **args)
            acc = f'{report["accuracy"]:.3f}'
            logger.info(f'{pl.i(dnm)} Classification Accuracy: {pl.i(acc)}')
            df = pd.DataFrame(report).transpose()
            df.to_csv(join(out_path, f'{dnm}.csv'))
