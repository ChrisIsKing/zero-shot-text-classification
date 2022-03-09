from typing import List, Dict

import json

from data_path import *


STSb = 'stsb_multi_mt'  # Per Hugging Face
config = {
    'fine-tune': dict(
        eg_sbert=dict(  # Per *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, section 4.2
            dataset_name=STSb,
            embedding_model_name='bert-base-uncased',
            max_seq_length=256,
            batch_size=16,
            n_epochs=4,
            n_eval=100,  # Total number of evaluations during training
            warmup_frac=0.1,
            pooling_model_kwargs=dict(pooling_mode_mean_tokens=True)
        )
    ),
    'datasets': {
        STSb: dict(
            n_sample=dict(
                train=5749,
                dev=1500,
                test=1379
            ),
            label_range=dict(
                min=0,
                max=5
            )
        )
    },
    'baselines': {
        'gpt2-nvidia': {
            'templates': [
                'To which category does the following document belong? : {}',
                'To which category does the following text belong? : {}',
                'To which category does the text belong? : {}',
                'To which category does the article belong? : {}',
                'How would you describe the following document? : as {}',
                'How would you describe the text? : as {}',
                'How would you describe the following text? : as {}',
                'Which best describes the text? : {}',
                'Which best describes the document? : {}',
                'Which best describes the following document? : {}',
                'Which best describes the following text? : {}',
                'The following document is _ ? : {}',
                'The following text is _ ? : {}',
                'The text is _ ? : {}',
                'The document is _ ? : {}',
                'How is the text best described? : {}',
                'How is the document best described? : {}',
                'How is the following text best described? : {}',
                'How is the following document best described? : {}',
                'Which of these choices best describes the text? : {}',
                'Which of these options best describes the text? : {}',
                'Which of these choices best describes the document? : {}',
                'Which of these options best describes the document? : {}',
                'Which of these categories best describes the following document? : {}',
                'Which of these choices best describes the following document? : {}',
                'Which of these options best describes the following text? : {}'
            ],
            'label-descriptors': dict(  # string label to natural language descriptor, as in paper
                ag_news={
                    'World': 'World News',
                    'Sports': 'Sports',
                    'Business': 'Business',
                    'Sci/Tech': 'Science & Technology'
                }
            )
        },
        'bert-mnli': dict(
            templates=dict(
                sentiment='This text expresses a {} sentiment',
                intent='This text expresses the intent of {}',
                topic='This text belongs to the topic of {}'
            )
        ),
    },
    'UTCD': dict(
        datasets=dict(
            # in-domain evaluation has the same labels as training
            clinc_150=dict(path='UTCD/clinc_150', aspect='intent', eval_labels_same=True, out_of_domain=False),
            # has some unique test labels
            sgd=dict(path='UTCD/sgd', aspect='intent', eval_labels_same=False, out_of_domain=False),
            slurp=dict(path='UTCD/slurp', aspect='intent', eval_labels_same=False, out_of_domain=False),
            emotion=dict(path='UTCD/emotion', aspect='sentiment', eval_labels_same=True, out_of_domain=False),
            go_emotion=dict(path='UTCD/go_emotion', aspect='sentiment', eval_labels_same=True, out_of_domain=False),
            sentiment_tweets_2020=dict(
                path='UTCD/sentiment_tweets_2020', aspect='sentiment', eval_labels_same=True, out_of_domain=False
            ),
            ag_news=dict(path='UTCD/ag_news', aspect='topic', eval_labels_same=True, out_of_domain=False),
            dbpedia=dict(path='UTCD/dbpedia', aspect='topic', eval_labels_same=True, out_of_domain=False),
            yahoo=dict(path='UTCD/yahoo', aspect='topic', eval_labels_same=True, out_of_domain=False),
            # Out-of-domain datasets: test split intended to evaluation
            amazon_polarity=dict(
                path='UTCD-ood/amazon_polarity', aspect='sentiment', eval_labels_same=True, out_of_domain=True
            ),
            finance_sentiment=dict(
                path='UTCD-ood/finance_sentiment', aspect='sentiment', eval_labels_same=True, out_of_domain=True
            ),
            yelp=dict(path='UTCD-ood/yelp', aspect='sentiment', eval_labels_same=True, out_of_domain=True),
            # Removed for too many options blow up GPT2's 1024 token length
            # arxiv=dict(path='UTCD-ood/arxiv', aspect='topic', eval_labels_same=True, out_of_domain=True),
            multi_eurlex=dict(path='UTCD-ood/multi_eurlex', aspect='topic', eval_labels_same=True, out_of_domain=True),
            patent=dict(path='UTCD-ood/patent', aspect='topic', eval_labels_same=True, out_of_domain=True),
            consumer_finance=dict(
                path='UTCD-ood/consumer_finance', aspect='topic', eval_labels_same=True, out_of_domain=True
            ),
            banking77=dict(path='UTCD-ood/banking77', aspect='intent', eval_labels_same=True, out_of_domain=True),
            snips=dict(path='UTCD-ood/snips', aspect='intent', eval_labels_same=True, out_of_domain=True),
            nlu_evaluation=dict(
                path='UTCD-ood/nlu_evaluation', aspect='intent', eval_labels_same=True, out_of_domain=True
            )
        ),
        dataset_ext='json'  # all in json
    ),
    'random-seed': 77
}

path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
ext = config['UTCD']['dataset_ext']


def path2dataset_labels(path: str) -> Dict[str, List[str]]:
    path = os.path.join(path_dset, f'{path}.{ext}')
    with open(path) as fl:
        dsets: Dict = json.load(fl)

    def samples2lbs(dset: List) -> List[str]:
        return sorted(set(lb for (txt, lb) in dset))  # Heuristic on how the `json` are stored
    return {split: samples2lbs(dset) for split, dset in dsets.items()}  # Labels for each split


d_dsets = config['UTCD']['datasets']
for dnm, d in d_dsets.items():
    d_labels = path2dataset_labels(d['path'])
    if d['eval_labels_same']:  # Sanity check
        assert d_labels['train'] == d_labels['test']
    d.update(dict(labels=d_labels))
dnms = sorted(d_dsets)  # All datasets, in- and out-of-domain, share the same dataset <=> id mapping
config['UTCD']['dataset_name2id'] = {dnm: i for i, dnm in enumerate(dnms)}
config['UTCD']['dataset_id2name'] = {i: dnm for i, dnm in enumerate(dnms)}


if __name__ == '__main__':
    from icecream import ic

    fl_nm = 'config.json'
    ic(config)
    with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
