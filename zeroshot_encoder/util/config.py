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
        }
    },
    'benchmark': dict(
        datasets=dict(
            clinc=dict(path='intent/clinc'),
            sgd=dict(path='intent/sgd'),
            slurp=dict(path='intent/slurp'),
            sentiment=dict(path='sentiment/emotion'),
            go_emotion=dict(path='sentiment/go_emotion'),
            sentiment_tweets_2020=dict(path='sentiment/sentiment_tweets_2020'),
            ag_news=dict(path='topic/ag_news'),
            dbpedia=dict(path='topic/dbpedia'),
            yahoo=dict(path='topic/yahoo')
        ),
        dataset_ext='json'  # all in json
    ),
    'random-seed': 77
}

path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
ext = config['benchmark']['dataset_ext']


def path2dataset_labels(path: str) -> Dict[str, List[str]]:
    path = os.path.join(path_dset, f'{path}.{ext}')
    with open(path) as fl:
        dsets: Dict = json.load(fl)

    def samples2lbs(dset: List) -> List[str]:
        return sorted(lb for (txt, lb) in dset)  # Heuristic on how the `json` are stored
    return {split: samples2lbs(dset) for split, dset in dsets.items()}  # Labels for each split


d_dsets = config['benchmark']['datasets']
for dnm, d in d_dsets.items():
    d.update(dict(labels=path2dataset_labels(d['path'])))
dnms = sorted(d_dsets)
config['benchmark']['dataset_name2id'] = {dnm: i for i, dnm in enumerate(dnms)}
config['benchmark']['dataset_id2name'] = {i: dnm for i, dnm in enumerate(dnms)}


if __name__ == '__main__':
    from icecream import ic

    fl_nm = 'config.json'
    ic(config)
    with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
