from transformers import AutoTokenizer
from dataset.load_data import get_all_zero_data

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)