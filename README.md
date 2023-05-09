# Zero-shot Text Classification

1. Benchmarking zero-shot text classification models
2. Bi-encoder for zero-shot classification, a balance between speed & accuracy.



## Universal Text Classification Dataset

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





## User’s Guide 

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



Below we include command line arguments and example train/eval commands. 





### BERT Sequence Classifier 

**Arguments** 

-   `dataset`: Dataset to train/evaluate the model on, pass `all` for all datasets 
-   `domain`: One of [`in`, `out`], the domain of dataset(s) to train/evaluate on 
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

-   Evaluate a model on out-of-domain dataset `multi_eurlex` 

    -   ```bash
        python zeroshot_classifier/models/bert.py test --domain out --dataset multi_eurlex --model_path models/2022-06-15_21-23-57_BERT-Seq-CLS-out-multi_eurlex/trained
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
-   `ddp`: DDP training flag, intended for proper training logging 
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





-   Train with learning rate 2e-5, ==TODO verify working== 

    -   ```bash
        python zeroshot_classifier/models/explicit/gpt2_pretrain.py --learning_rate 4e-5 output_dir '{a=4e-5}'
        ```



