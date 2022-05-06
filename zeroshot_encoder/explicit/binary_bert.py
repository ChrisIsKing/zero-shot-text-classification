import math
import random
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from os.path import join
from typing import List, Type, Dict, Callable, Optional, Union, Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizer
from transformers import AdamW, get_scheduler
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from os.path import join
from sklearn.metrics import classification_report
from zeroshot_encoder.util.load_data import get_data, binary_explicit_format, in_domain_data_path, out_of_domain_data_path
from stefutil import *
from zeroshot_encoder.util.util import sconfig
from zeroshot_encoder.util import *

logger = logging.getLogger(__name__)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class BertZeroShotExplicit(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.binary_cls = nn.Linear(self.config.hidden_size, 2)
        self.aspect_cls = nn.Linear(self.config.hidden_size, 3)

        # Initialize weights and apply final processing
        self.post_init()
    
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
        aspect_logits = self.aspect_cls(pooled_output)
        
        loss = None
        
        logits = {'cls': binary_logits, 'aspect': aspect_logits}
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits, outputs.hidden_states, outputs.attentions


class ExplicitCrossEncoder:
    def __init__(self, name="bert-base-uncased", device="cuda", max_length=None) -> None:
        self.config = BertConfig.from_pretrained(name)
        self.model = BertZeroShotExplicit(self.config)
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.device = device
        self.max_length = max_length
    
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
        aspects = torch.tensor(labels, dtype=torch.long).to(self.device)

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

    def fit(self,
        train_dataloader: DataLoader,
        epochs: int = 1,
        scheduler: str = 'linear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = AdamW,
        optimizer_params: Dict[str, object] = {'lr': 2e-5},
        weight_decay: float = 0.01,
        output_path: str = None,
        max_grad_norm: float = 1,
        show_progress_bar: bool = True):

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

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels, aspects in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                model_predictions = self.model(**features, return_dict=True)

                pooled_output = model_predictions[1]
                loss = None
                loss_fct = CrossEntropyLoss()

                task_loss_value = loss_fct(pooled_output['aspect'], aspects)
                binary_loss_value = loss_fct(pooled_output['cls'], labels)
                loss = task_loss_value + binary_loss_value
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()

            lr_scheduler.step()

            training_steps += 1
    
        self.save(output_path)

    def predict(self, sentences: List[List[str]], batch_size: int = 32):
        
        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, shuffle=False)

        show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

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

                
                logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

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
                        default=0.00002,
                        type=float,
                        help="The initial learning rate for Adam.")

    # set test arguments
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.command == 'train':
        device = torch.device("cuda")

        data = get_data(in_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())
        train = binary_explicit_format(data)

        train_batch_size = args.batch_size
        lr = args.learning_rate
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

        model = ExplicitCrossEncoder("bert-base-uncased", device=device)

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

        model.fit(
            train_dataloader=train_dataloader,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params= {'lr': lr},
            output_path=model_save_path)
    
    if args.command == 'test':
        mode = args.mode
        pred_path = join(args.model_path, 'preds/{}/'.format(args.domain))
        result_path = join(args.model_path, 'results/{}/'.format(args.domain))
        Path(pred_path).mkdir(parents=True, exist_ok=True)
        Path(result_path).mkdir(parents=True, exist_ok=True)
        if args.domain == 'in':
            data = get_data(in_domain_data_path)
        else:  # out
            data = get_data(out_of_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())

        model = ExplicitCrossEncoder(args.model_path)

        label_map = ["false", "true"]

        # loop through all datasets
        for dataset in datasets:
            examples = data[dataset]["test"]
            labels = data[dataset]['labels']
            aspect = data[dataset]['aspect']
            preds = []
            gold = []
            correct = 0

            if mode == 'vanilla':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, lb] for lb in lbs]
            elif mode == 'implicit':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, f'{lb} {aspect}'] for lb in lbs]
            elif mode == 'implicit-on-text-encode-aspect':
                aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')[aspect]

                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect_token} {txt}', lb] for lb in lbs]
            elif mode == 'implicit-on-text-encode-sep':
                sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')

                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect} {sep_token} {txt}', lb] for lb in lbs]
            else:
                raise NotImplementedError(f'{logi(mode)} not supported yet')

            # loop through each test example
            print(f'Evaluating dataset: {logi(dataset)}')
            for index, (text, gold_labels) in enumerate(tqdm(examples.items())):
                query = txt_n_lbs2query(text, labels)
                results = model.predict(query)

                # compute which pred is higher
                pred = labels[results[:, 1].argmax()]
                preds.append(pred)
               
                if pred in gold_labels:
                    correct += 1
                    gold.append(pred)
                else:
                    gold.append(gold_labels[0])
            
            print(f'{logi(dataset)} Dataset Accuracy: {logi(correct/len(examples))}')
            report = classification_report(gold, preds, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv('{}/{}.csv'.format(result_path, dataset))