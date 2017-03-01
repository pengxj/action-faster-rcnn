# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick 

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.ucfsports import ucfsports
from datasets.JHMDB import JHMDB
from datasets.UCF101 import UCF101
# import datasets.MSRII
import numpy as np

DATA_LIST = ['UCF-Sports', 'JHMDB', 'UCF101', 'MSRII']
MOD_LIST = ['RGB', 'FLOW']
LEN_LIST = ['1', '5', '10']
SPLIT_LIST = ['0', '1', '2']
for dataset in DATA_LIST:
    for mod in MOD_LIST:
        for lens in LEN_LIST:
            for split in SPLIT_LIST:
                name = '{}_{}_{}_split_{}'.format(dataset, mod, lens, split)
                image_set = name
                if dataset is 'UCF-Sports':
                    __sets[name] = (lambda split=image_set, phase='TRAIN': ucfsports(split, phase))
                elif dataset is 'JHMDB':
                    __sets[name] = (lambda split=image_set, phase='TRAIN': JHMDB(split, phase)) 
                elif dataset is 'UCF101':
                    __sets[name] = (lambda split=image_set, phase='TRAIN': UCF101(split, phase))
                # elif dataset is 'MSRII':
                #     __sets[name] = (lambda split=split, datapath='/home/lear/xpeng/data/MSR_II/pweinzaeMSR2/frames': #frames #features/fat2/motion_cnn_proposals/jpeg0
                #             datasets.MSRII(split, datapath))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
