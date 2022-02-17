# Zero-shot Bi-encoder 
1. Benchmarking zero-shot text classification models
2. Bi-encoder for zero-shot classification, a balance between speed & accuracy.




## To Use 
Python version `3.9.7`.

```bash
$ pip3 install -r requirements.txt
```


### Train Baseline 
e.g. On GPT2 zero shot classification: 
```bash
$ export PYTHONPATH=$PATHONPATH:`pwd`
$ python zeroshot_encoder/baseline/gpt2.py
```


## Obsolete: Unified-Encoder
Exploring a unified framework for potentially many NLP tasks as encoding operations



Formalize common NLP tasks beyond Information Retrieval as 1) encoding then 2) simple operation, so that intermediate results can be cached. 

