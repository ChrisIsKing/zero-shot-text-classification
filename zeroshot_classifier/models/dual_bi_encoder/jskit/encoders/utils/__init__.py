# ========================== Begin of modified ==========================
# from .models import *
# from .train import *
# from .evaluate import *
# from .tokenizer import *
# Doesn't match with import usage in `bi.py`
import os
from zeroshot_classifier.util import BASE_PATH, PROJ_DIR, PKG_NM
CONFIG_PATH = os.path.join(
    BASE_PATH, PROJ_DIR, PKG_NM, 'models',
    'dual_bi_encoder', 'jskit', 'encoders', 'utils', 'config.cfg'
)

from . import models
from . import tokenizer
from . import train
from . import evaluate
# ========================== End of modified ==========================

