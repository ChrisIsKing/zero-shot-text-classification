import math
import logging
import os
import random
from statistics import mode
from torch import ge
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from dataset.load_data import get_all_zero_data, binary_cls_format

logger = logging.getLogger(__name__)

data = get_all_zero_data()
train = binary_cls_format(data["all"]["train"])
test = binary_cls_format(data["all"]["test"], train=False)

train_batch_size = 16
num_epochs = 3
model_save_path = 'models/binary_bert'

model = CrossEncoder('bert-base-uncased', num_labels=2)

train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test, name='UTCD-test')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)