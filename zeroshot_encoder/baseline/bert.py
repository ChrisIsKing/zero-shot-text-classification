import math
import logging
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
from zeroshot_encoder.util.load_data import get_data, seq_cls_format, in_domain_data_path, out_of_domain_data_path

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()

    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    parser_train.add_argument('--dataset', type=str, required=True)
    parser_train.add_argument('--domain', type=str, choices=['in', 'out'], required=True)

    parser_test.add_argument('--dataset', type=str, required=True)
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--path', type=str, required=True)

    return parser.parse_args()


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }



if __name__ == "__main__":
    args = parse_args()

    if args.command == 'train':
        if args.dataset:

            if args.domain == "in":
                data = get_data(in_domain_data_path)
            else:
                data = get_data(out_of_domain_data_path)

            if args.dataset == "all":
                dataset = data
                train, test, labels = seq_cls_format(dataset, all=True)
            else:
                # get dataset data
                dataset = data[args.dataset]
                train, test, labels = seq_cls_format(dataset)
            
            num_labels = len(labels)

            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True, num_labels=num_labels)
            model.to("cuda")

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)


            train_dataset = Dataset.from_pandas(pd.DataFrame(train))
            test_dataset = Dataset.from_pandas(pd.DataFrame(test))

            train_dataset = train_dataset.map(tokenize_function, batched=True)
            test_dataset = test_dataset.map(tokenize_function, batched=True)

            output_path = './models/{}'.format(args.dataset)
            num_epochs = 3
            warmup_steps = math.ceil(len(train_dataset) * num_epochs * 0.1) #10% of train data for warm-up
            logger.info("Warmup-steps: {}".format(warmup_steps))

            training_args = TrainingArguments(
                output_dir=output_path,          # output directory
                num_train_epochs=num_epochs,     # total number of training epochs
                per_device_train_batch_size=16,  # batch size per device during training
                per_device_eval_batch_size=20,   # batch size for evaluation
                warmup_steps=warmup_steps,       # number of warmup steps for learning rate scheduler
                weight_decay=0.01,               # strength of weight decay
                logging_dir='./logs',            # directory for storing logs
                load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
                # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
                logging_steps=100000,               # log & save weights each logging_steps
                save_steps=100000,
                evaluation_strategy="steps",     # evaluate each `logging_steps`
            )

            trainer = Trainer(
                model=model,                         # the instantiated Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=test_dataset,          # evaluation dataset
                compute_metrics=compute_metrics,     # the callback that computes metrics of interest
            )

            trainer.train()
            print(trainer.evaluate())
    
    if args.command == 'test':

        if args.domain == "in":
            data = get_data(in_domain_data_path)
        else:
            data = get_data(out_of_domain_data_path)
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(args.path)
        model.to("cuda")

        def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = data[args.dataset]
        train, test, labels = seq_cls_format(dataset)
        test_dataset = Dataset.from_pandas(pd.DataFrame(test))
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        output_path = './models/{}'.format(args.dataset)

        training_args = TrainingArguments(
            output_dir=output_path,          # output directory
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
            logging_steps=100000,               # log & save weights each logging_steps
            save_steps=100000,
            evaluation_strategy="steps",     # evaluate each `logging_steps`
        )

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        )

        print(trainer.evaluate())