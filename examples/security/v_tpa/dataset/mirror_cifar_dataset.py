#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

from .cifar_dataset import CIFARDataset
from .badnets_base_dataset import BadNetsBaseDataset

class MirrorCIFARDataset(CIFARDataset, BadNetsBaseDataset):
    def __init__(self, dataset_name, data_path, args={}, mirror_args={}):
        CIFARDataset.__init__(self, dataset_name, data_path, args)
        BadNetsBaseDataset.__init__(self, self.train_dataset, self.valid_dataset, mirror_args)
