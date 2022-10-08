import os

__all__ = ['BASE_PATH', 'PROJ_DIR', 'PKG_NM', 'DSET_DIR', 'MODEL_DIR']

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

# Absolute system path for root directory;
#   e.g.: '/Users/stefanh/Documents/UMich/Research/Clarity Lab/Zeroshot Text Classification'
BASE_PATH = os.sep.join(paths[:-2])  # System data path
# Repo root folder name with package name; e.g.: 'Zeroshot-Text-Classification'
PROJ_DIR = paths[-2]
PKG_NM = paths[-1]  # Package/Module name, e.g. `zeroshot_classifier`

MODEL_DIR = 'models'  # Save models
DSET_DIR = 'dataset'


if __name__ == '__main__':
    from stefutil import mic
    mic(BASE_PATH, type(BASE_PATH), PROJ_DIR, PKG_NM)
