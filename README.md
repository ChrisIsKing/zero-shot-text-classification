# Label Agnostic Pre-training for Zero-shot Text Classification

This repository contains the code and data for the Findings of ACL'23 paper **Label Agnostic Pre-training for Zero-shot Text Classification** by ***Christopher Clarke, Yuzhao Heng, Yiping Kang, Krisztian Flautner, Lingjia Tang and Jason Mars***. 

In this paper, we investigate the task of zero-shot text classification with the aim of improving the ability of PLMs to generalize both seen and unseen data across domains without the need for additional training. We introduce two new simple yet effective training strategies, *Implicit training* & *Explicit pre-training* which specifically inject aspect-level understanding into the model at train time. To evaluate this, we release UTCD, a new benchmark dataset for evaluating text classification in zero-shot settings. **Models, data & paper coming soon!**

## Universal Text Classification Dataset (UTCD)
UTCD is a compilation of 18 classification datasets spanning 3 categories of Sentiment, Intent/Dialogue and Topic classification. UTCD focuses on the task of zero-shot text classification where the candidate labels are descriptive of the text being classified. UTCD consists of ~ 6M/800K train/test examples.

UTCD Datasets & Principles:

- Sentiment
    - GoEmotions introduced in [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/pdf/2005.00547v2.pdf)
    - TweetEval introduced in [TWEETEVAL: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/pdf/2010.12421v2.pdf) (Sentiment subset)
    - Emotion introduced in [CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404.pdf)
    - Amazon Polarity introduced in [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)
    - Finance Phrasebank introduced in [Good debt or bad debt: Detecting semantic orientations in economic texts](https://arxiv.org/pdf/1307.5336.pdf)
    - Yelp introduced in [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)
- Intent/Dialogue
    - Schema-Guided Dialogue introduced in [Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/1909.05855v2.pdf)
    - Clinc-150 introduced in [An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://arxiv.org/pdf/1909.02027v1.pdf)
    - SLURP SLU introduced in [SLURP: A Spoken Language Understanding Resource Package](https://arxiv.org/pdf/2011.13205.pdf)
    - Banking77 introduced in [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/pdf/2003.04807.pdf)
    - Snips introduced in [Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces](https://arxiv.org/pdf/1805.10190.pdf)
    - NLU Evaluation introduced in [Benchmarking Natural Language Understanding Services for building Conversational Agents](https://arxiv.org/pdf/1903.05566.pdf)
- Topic
    - AG News introduced in [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)
    - DBpedia 14 introduced in [DBpedia: A Nucleus for a Web of Open Data](https://link.springer.com/chapter/10.1007/978-3-540-76298-0_52)
    - Yahoo Answer Topics introduced in [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)
    - MultiEurlex introduced in [MultiEURLEX -- A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer](https://aclanthology.org/2021.emnlp-main.559v2.pdf)
    - BigPatent introduced in [BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization](https://aclanthology.org/P19-1212.pdf)
    - Consumer Finance introduced in [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

In order to make NLP models more broadly useful, zero-shot techniques need to be capable of label, domain \& aspect transfer. As such, in the construction of UTCD we enforce the following principles: 

- **Textual labels**: In UTCD, we mandate the use of textual labels. While numerical label values are often used in classification tasks, descriptive textual labels such as those present in the datasets across UTCD enable the development of techniques that can leverage the class name which is instrumental in providing zero-shot support. As such, for each of the compiled datasets, labels are standardized such that the labels are descriptive of the text in natural language. 
- **Diverse domains and Sequence lengths**: In addition to broad coverage of aspects, UTCD compiles diverse data across several domains such as Banking, Finance, Legal, etc each comprising varied length sequences (long and short). The datasets are listed above.

## User’s Guide (HuggingFace)

The [UTCD dataset](https://huggingface.co/datasets/claritylab/UTCD) and [trained models](https://huggingface.co/models?other=zeroshot_classifier) are available on HuggingFace. Please refer to the instructions there.



## User’s Guide (Local)

### Setup environment

OS: UNIX; Python version `3.8.10`; CUDA version `11.6`. 



Create conda environment: 

```bash
conda create -n zs-cls python=3.8.10 pip
```

Move to project root directory, install python packages: 

```bash
pip3 install -r requirements.txt
```

Add current directory for python to look for our local package: 

```bash
export PYTHONPATH=$PATHONPATH:`pwd`
```



### Note 

Denoting the package directory at system path `<BASE_PATH>/zero-shot-text-classification`, all trained models will be saved to `<BASE_PATH>/models`, all evaluation CSV files will be saved to `<BASE_PATH>/eval`. 



Below we include command line arguments and example train/eval commands for models in our paper. 





### BERT Sequence Classifier 

**Arguments** 

-   `dataset`: Dataset to train/evaluate the model on, pass `all` for all datasets 
-   `domain`: One of [`in`, `out`], the domain of dataset(s) to train/evaluate on 
-   `normalize_aspect`: If true, datasets are normalized by aspect, ==TODO add== 
-   `learning_rate`: Learning rate for training 
-   `batch_size`: Batch size for training/evaluation  
-   `epochs`: #epochs for training 
-   `model_name_or_path`: File system path or HuggingFace model name for model evaluation, ==TODO test== 





**Train** 

-   Train solely on in-domain dataset `go_emotion`

    -   ```bash
        python zeroshot_classifier/models/bert.py train --domain in --dataset go_emotion
        ```

-   Train solely on out-of-domain dataset `consumer_finance` 

    -   ```bash
        python zeroshot_classifier/models/bert.py train --domain out --dataset consumer_finance
        ```

-   Train on all in-domain datasets 

    -   ```bash
        python zeroshot_classifier/models/bert.py train --domain in --dataset all
        ```





**Eval**

-   Evaluate a local model on out-of-domain dataset `multi_eurlex` 

    -   ```bash
        python zeroshot_classifier/models/bert.py test --domain out --dataset multi_eurlex --model_name_or_path models/2022-06-15_21-23-57_BERT-Seq-CLS-out-multi_eurlex/trained
        ```
    
    





### Binary & Dual Encoding Zero-shot Classification

**Arguments** 

-   `mode`: Training strategy, one of [`vanilla`, `implicit-on-text-encode-sep`, `explicit`] 
-   `normalize_aspect`: If true, datasets are normalized by aspect, ==TODO add== 
-   `learning_rate`: Learning rate for training 
-   `batch_size`: Batch size for training/evaluation  
-   `epochs`: #epochs for training 
-   `init_model_name_or_path`: Fie system path or HuggingFace model name to initialize model weights for explicit training, ==TODO test== 
-   `output_dir`: Directory name postfix for trained model 
-   `domain`: One of [`in`, `out`], the domain of datasets to evaluate on 
-   `model_name_or_path`: Directory name or HuggingFace model name for evaluation 





**Train**

-   Vanilla training on Binary BERT 

    -   ```bash
        python zeroshot_classifier/models/binary_bert.py train --mode vanilla --batch_size 32 --epochs 8 --learning_rate 2e-5 --output_dir '{a=2e-5}'
        ```

-   Explicit training on Bi-Encoder 

    -   ```bash
        python zeroshot_classifier/models/bi-encoder.py train --mode explicit --model_init '2022-11-21_18-58-54_Aspect-Pretrain-Binary-BERT_{md=exp, na=T}_{a=3e-05}/trained'
        ```





**Eval**

-   Evaluate implicitly-trained model on all in-domain datasets 

    -   ```bash
        python zeroshot_classifier/models/binary_bert.py test --mode implicit-on-text-encode-sep --domain in --model_dir_nm 2022-10-12_01-21-08_Binary-BERT-implicit-on-text-encode-sep-rand-aspect-norm
        ```





#### Explicit Pretraining

**Arguments** 

-   `output_dir`: Directory name postfix for trained model 
-   `normalize_aspect`: If true, datasets are normalized by aspect 
-   `learning_rate`: Learning rate for training 
-   `batch_size`: Batch size for training/evaluation 
-   `epochs`: #epochs for training 





**Train**

-   Train with learning rate 2e-5, ==TODO verify working== 

    -   ```bash
        python zeroshot_classifier/models/explicit/binary_bert_pretrain.py --learning_rate 2e-5 output_dir '{a=2e-5}'
        ```





### Generative Classification

**Arguments**

-   `mode`: Training strategy, one of [`vanilla`, `implicit`, `explicit`] 
-   `normalize_aspect`: If true, datasets are normalized by aspect 
-   `learning_rate`: Learning rate for training 
-   `batch_size`: Batch size for training/evaluation  
-   `gradient_accumulation_steps`: #gradient accumulation steps for training 
-   `epochs`: #epochs for training 
-   `ddp`: DDP training flag, intended for proper logging during training
-   `init_model_name_or_path`: Fie system path or HuggingFace model name to initialize model weights for explicit training, ==TODO verify working== 
-   `output_dir`: Directory name postfix for trained model 
-   `model_name_or_path`: Directory name for model evaluation 





==TODO, verify command args==

**Train** 

-   Implicit training on GPT with DDP 

    -   ```bash
        torchrun --nproc_per_node=4 zeroshot_classifier/models/gpt2.py train --mode implicit
        ```

-   Explicit training on GPT 

    -   ```bash
        python zeroshot_classifier/models/gpt2.py train --mode explicit --model_init '2022-11-27_17-39-06_Aspect-Pretrain-NVIDIA-GPT2_{md=exp, na=T}_{a=2e-05}'
        ```





**Eval**

-   Evaluate model with vanilla training on all out-of-domain datasets 

    -   ```bash
        python zeroshot_classifier/models/gpt2.py test --mode implicit --model_dir_nm '2022-11-29_19-37-13_NVIDIA-GPT2_{md=van, na=T}_{a=3e-05}'
        ```





#### Explicit Pretraining

**Arguments** 

-   `output_dir`: Directory name postfix for trained model 
-   `normalize_aspect`: If true, datasets are normalized by aspect 
-   `learning_rate`: Learning rate for training 
-   `batch_size`: Batch size for training/evaluation 
-   `gradient_accumulation_steps`: #gradient accumulation steps for training 
-   `epochs`: #epochs for training 





**Train**

-   Train with learning rate 2e-5, ==TODO verify working== 

    -   ```bash
        python zeroshot_classifier/models/explicit/gpt2_pretrain.py --learning_rate 4e-5 output_dir '{a=4e-5}'
        ```



