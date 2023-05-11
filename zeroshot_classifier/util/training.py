from typing import NamedTuple, Tuple, Union

import numpy as np


__all__ = ['PT_LOSS_PAD', 'MyEvalPrediction']


PT_LOSS_PAD = -100  # Pytorch indicator value for ignoring loss, used in huggingface for padding tokens


class MyEvalPrediction(NamedTuple):
    """
    Support `dataset_id`, see `compute_metrics` and `CustomTrainer.prediction_step`
    """
    # wouldn't work if subclass `EvalPrediction`, see https://github.com/python/mypy/issues/11721
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    dataset_ids: Union[np.ndarray, Tuple[np.ndarray]]
