import os

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

# Absolute system path for root directory;
#   e.g.: '/Users/stefanh/Documents/UMich/Research/Clarity Lab/Unified Encoder'
PATH_BASE = os.sep.join(paths[:-2])  # System data path
# Repo root folder name with package name; e.g.: ''Unified-Encoder/unified-encoder''
DIR_PROJ = paths[-2]
PKG_NM = paths[-1]  # Package/Module name

DIR_MDL = 'models'  # Save models
DIR_DSET = 'dataset'


if __name__ == '__main__':
    from icecream import ic
    ic(PATH_BASE, type(PATH_BASE), DIR_PROJ)
