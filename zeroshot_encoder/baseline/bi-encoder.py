import math
import random
import logging
import json
import pandas as pd
from numpy import argmax
from zeroshot_encoder.util.load_data import get_data, binary_cls_format, in_domain_data_path, out_of_domain_data_path
from os.path import join
from sentence_transformers import SentenceTransformer, models,losses, evaluation, util
from torch import nn
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report

def parse_args():

    parser =  ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    train = subparser.add_parser('train')
    test = subparser.add_parser('test')

    # set train arguments
    train.add_argument('--output', type=str, required=True)
    train.add_argument('--sampling', type=str, choices=['rand', 'vect'], required=True)
    train.add_argument('--mode', type=str, choices=['vanilla', 'implicit', 'explicit'], default='vanilla')
    train.add_argument('--batch_size', type=int, default=16)
    train.add_argument('--epochs', type=int, default=3)
    

    # set test arguments
    test.add_argument('--model_path', type=str, required=True)
    test.add_argument('--domain', type=str, choices=['in', 'out'] ,required=True)
    
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
            if args.mode == 'vanilla':
                train += binary_cls_format(data[dataset], name=dataset, sampling=args.sampling)
                test += binary_cls_format(data[dataset], train=False)
        
        train_batch_size = args.batch_size
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
        # Add end of turn token for sgd
        word_embedding_model.tokenizer.add_special_tokens({'eos_token': '[eot]'})
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        random.shuffle(train)

        train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

        evauator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test, name='UTCD-test')

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

        model.fit(train_objectives=[(train_dataloader, train_loss)], 
                epochs=num_epochs,
                evaluator=evauator, 
                warmup_steps=warmup_steps,
                evaluation_steps=100000,
                output_path=model_save_path)
    
    if args.command == 'test':
        pred_path = join(args.model_path, 'preds/{}/'.format(args.domain))
        result_path = join(args.model_path, 'results/{}/'.format(args.domain))
        Path(pred_path).mkdir(parents=True, exist_ok=True)
        Path(result_path).mkdir(parents=True, exist_ok=True)
        if args.domain == 'in':
            data = get_data(in_domain_data_path)
        elif args.domain == 'out':
            data = get_data(out_of_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())

        # load model
        model = SentenceTransformer(args.model_path)

        # loop through all datasets
        for dataset in datasets:
            examples = data[dataset]["test"]
            labels = data[dataset]['labels']
            preds = []
            gold = []
            correct = 0

            example_vectors = model.encode(list(examples.keys()))
            label_vectors = model.encode(labels)
            
            # loop through each test example
            print("Evaluating dataset: {}".format(dataset))
            for index, (text, gold_labels) in enumerate(tqdm(examples.items())):
                results = [util.cos_sim(example_vectors[index], label_vectors[i]) for i in range(len(labels))]

                # compute which pred is higher
                pred = labels[argmax(results)]
                preds.append(pred)
               
                if pred in gold_labels:
                    correct += 1
                    gold.append(pred)
                else:
                    gold.append(gold_labels[0])
            
            print('{} Dataset Accuracy = {}'.format(dataset, correct/len(examples)))
            report = classification_report(gold, preds, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv('{}/{}.csv'.format(result_path, dataset))