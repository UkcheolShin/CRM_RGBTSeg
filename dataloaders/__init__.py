# Written by Ukcheol Shin, Jan. 24, 2023
# Email: shinwc159@gmail.com

from .MF_dataset import MF_dataset
from .KP_dataset import KP_dataset
from .PST_dataset import PST_dataset
from .augmentation import *

def build_dataset(cfg):
    """
    Return corresponding dataset according to given dataset option
    :param config option
    :return Dataset class
    """

    # Set dataset
    dataset_name = cfg.DATASETS.NAME
    dataset={}

    if dataset_name == 'MFdataset':
        dataset['train'] = MF_dataset(cfg.DATASETS.DIR, cfg, split='train')
        dataset['val']   = MF_dataset(cfg.DATASETS.DIR, cfg, split='val')
        dataset['test']  = MF_dataset(cfg.DATASETS.DIR, cfg, split='test')
    elif dataset_name == 'KPdataset':
        dataset['train'] = KP_dataset(cfg.DATASETS.DIR, cfg, split='train')
        dataset['val']   = KP_dataset(cfg.DATASETS.DIR, cfg, split='val')
        dataset['test']  = KP_dataset(cfg.DATASETS.DIR, cfg, split='test')
    elif dataset_name == 'PSTdataset': # PST900 dataset doesn't has validation set
        dataset['train'] = PST_dataset(cfg.DATASETS.DIR, cfg, split='train')
        dataset['val']   = PST_dataset(cfg.DATASETS.DIR, cfg, split='test')
        dataset['test']  = PST_dataset(cfg.DATASETS.DIR, cfg, split='test')
    else:
        raise ValueError('Unknown dataset type: {}.'.format(dataset_name))

    return dataset
