
from argparse import ArgumentParser

from zeroshot_classifier.util import *


__all__ = ['HF_MODEL_NAME', 'parse_args']


HF_MODEL_NAME = 'bert-base-uncased'


def parse_args():
    modes = sconfig('training.strategies')

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--output', type=str, default=None)
    parser_train.add_argument('--output_dir', type=str, default=None)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], default='rand')
    parser_train.add_argument('--normalize_aspect', type=bool, default=True)
    # model to initialize weights from, intended for loading weights from local explicit training
    parser_train.add_argument('--init_model_name_or_path', type=str, default=HF_MODEL_NAME)
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--learning_rate', type=float, default=2e-5)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)

    # set test arguments
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_test.add_argument('--batch_size', type=int, default=32)  # #of texts to do inference in a single forward pass
    parser_test.add_argument('--model_name_or_path', type=str, required=True)

    return parser.parse_args()
