import math
import random
import logging
from typing import List
from pathlib import Path
from os.path import join
from argparse import ArgumentParser

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from tqdm import tqdm

from stefutil import *
from zeroshot_encoder.util import *
from zeroshot_encoder.util.load_data import get_data, binary_cls_format, in_domain_data_path, out_of_domain_data_path


random.seed(42)  # for negative sampling


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

    # set test arguments
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    
    return parser.parse_args()


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    args = parse_args()
    if args.command == 'train':
        data = get_data(in_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())
        train = []
        test = []
        for dataset in datasets:
            train += binary_cls_format(data[dataset], name=dataset, sampling=args.sampling, mode=args.mode)
            test += binary_cls_format(data[dataset], train=False, mode=args.mode)

        train_batch_size = args.batch_size
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        model = CrossEncoder('bert-base-uncased', num_labels=2)
        spec_tok_args = dict(eos_token='[eot]')  # Add end of turn token for sgd
        add_spec_toks = None
        if args.mode == 'implicit-on-text-encode-aspect':
            add_spec_toks = list(sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token').values())
        elif args.mode == 'implicit-on-text-encode-sep':
            add_spec_toks = [sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')]
        if add_spec_toks:
            spec_tok_args |= dict(additional_special_tokens=add_spec_toks)
        model.tokenizer.add_special_tokens(spec_tok_args)
        model.model.resize_token_embeddings(len(model.tokenizer))

        train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test, name='UTCD-test')

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=100000,
            warmup_steps=warmup_steps,
            output_path=model_save_path
        )
    if args.command == 'test':
        mode = args.mode
        ca(domain=args.domain)
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

        # load model
        model = CrossEncoder(args.model_path)

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
                results = model.predict(query, apply_softmax=True)

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
