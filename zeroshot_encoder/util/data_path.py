import os

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

# Absolute system path for root directory;
#   e.g.: '/Users/stefanh/Documents/UMich/Research/Clarity Lab/Zeroshot Text Classification'
PATH_BASE = os.sep.join(paths[:-2])  # System data path
# Repo root folder name with package name; e.g.: 'Zeroshot-Text-Classification'
DIR_PROJ = paths[-2]
PKG_NM = paths[-1]  # Package/Module name, e.g. `zeroshot_encoder`

DIR_MDL = 'models'  # Save models
DIR_DSET = 'dataset'


if __name__ == '__main__':
    from icecream import ic
    ic(PATH_BASE, type(PATH_BASE), DIR_PROJ, PKG_NM)
