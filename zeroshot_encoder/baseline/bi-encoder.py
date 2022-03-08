import math
import random
import logging
import json
import pandas as pd
from numpy import argmax
from zeroshot_encoder.util.load_data import get_data, encoder_cls_format, in_domain_data_path, out_of_domain_data_path
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
        datasets.remove("all")
        train = []
        test = []
        for dataset in datasets:
            train += encoder_cls_format(data[dataset]["train"], name=dataset, sampling=args.sampling)
            test += encoder_cls_format(data[dataset]["test"], name=dataset, train=False)
        
        train_batch_size = args.batch_size
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
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
        pred_path = join(args.model_path, 'preds')
        result_path = join(args.model_path, 'results')
        Path(pred_path).mkdir(parents=True, exist_ok=True)
        Path(result_path).mkdir(parents=True, exist_ok=True)
        if args.domain == 'in':
            data = get_data(in_domain_data_path)
        elif args.domain == 'out':
            data = get_data(out_of_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())
        datasets.remove("all")

        # load model
        model = SentenceTransformer(args.model_path)

        # label_map = ["false", "true"]

        # loop through all datasets
        for dataset in datasets:
            test = data[dataset]["test"]
            count = Counter([x[0] for x in test])
            gold = []
            examples = []
            labels = list(dict.fromkeys(x[1] for x in test))
            label_vectors = model.encode(labels)

            # Deal w/multi-label & duplicates
            duplicates = [k for k,v in count.items() if v > 1]
            for duplicate in duplicates:
                examples.append(duplicate)
                
                # get labels for duplicate
                dup_labels = [x[1] for x in test if x[0] == duplicate]
                gold.append([1 if label in dup_labels else 0 for label in labels])

            for x, y in test:
                if count[x] > 1:
                    continue
                else:
                    examples.append(x)
                    gold.append([1 if y==label else 0 for label in labels])
            
            assert len(examples) == len(gold)

            example_vectors = model.encode(examples)
            
            preds = []
            gold_labels = []
            correct = 0
            # loop through each test example
            print("Evaluating dataset: {}".format(dataset))
            for index, example in enumerate(tqdm(examples)):
                results = [util.cos_sim(example_vectors[index], label_vectors[i]) for i in range(len(labels))]

                # compute which pred is higher
                pred = labels[argmax(results)]
                preds.append(pred)
                # load gold labels
                g_label = [labels[i] for i, l in enumerate(gold[index]) if l==1]
                if pred in g_label:
                    correct += 1
                    gold_labels.append(pred)
                else:
                    gold_labels.append(g_label[0])
                
            
            print('{} Dataset Accuracy = {}'.format(dataset, correct/len(examples)))
            report = classification_report(gold_labels, preds, output_dict=True)
            json.dump([ [examples[i], pred, gold_labels[i]] for i, pred in enumerate(preds)], open('{}/{}.json'.format(pred_path,dataset), 'w'), indent=4)
            # plt = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
            # plt.figure.savefig('figures/binary_bert_{}.png'.format(dataset))
            df = pd.DataFrame(report).transpose()
            df.to_csv('{}/{}.csv'.format(result_path, dataset))