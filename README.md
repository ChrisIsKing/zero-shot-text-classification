# Zero-shot Bi-encoder 
1. Benchmarking zero-shot text classification models
2. Bi-encoder for zero-shot classification, a balance between speed & accuracy.


## To Use 
Python version `3.8.10`.

```bash
pip3 install -r requirements.txt
```
### Universal Text Classification Dataset
UTCD is a compilation of 9 classification datasets spanning 3 categories of Sentiment, Intent/Dialogue and Topic classification. UTCD focuses on the task of zero-shot text classification where the candidate labels are descriptive of the text being classified. UTCD consists of ~ 2.3M/200K train/test examples and can be downloaded [here](https://drive.google.com/file/d/1qISYYoQNGXtmGWrCsKoK-fBKt8MHXqR7/view?usp=sharing)

UTCD Datasets:

- Sentiment
    - GoEmotions dataset introduced in [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/pdf/2005.00547v2.pdf)
    - TweetEval dataset introduced in [TWEETEVAL: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/pdf/2010.12421v2.pdf) (Sentiment subset)
    - Emotion dataset introduced in [CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404.pdf)
- Intent/Dialogue
    - Schema-Guided Dialogue dataset introduced in [Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/1909.05855v2.pdf)
    - Clinc-150 introduced in [An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://arxiv.org/pdf/1909.02027v1.pdf)
    - SLURP SLU dataset introduced in [SLURP: A Spoken Language Understanding Resource Package](https://arxiv.org/pdf/2011.13205.pdf)
- Topic
    - AG News introduced in [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)
    - DBpedia 14 introduced in [DBpedia: A Nucleus for a Web of Open Data](https://link.springer.com/chapter/10.1007/978-3-540-76298-0_52)
    - Yahoo Answer Topics introduced in [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)

### Train Baseline 
e.g. On GPT2 zero shot classification: 
```bash
export PYTHONPATH=$PATHONPATH:`pwd`
python zeroshot_encoder/baseline/gpt2.py
```

