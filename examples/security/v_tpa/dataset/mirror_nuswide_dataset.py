#!/usr/bin/env python
# coding=utf-8
from .badnets_base_dataset import BadNetsBaseDataset
from .nuswide_dataset import NUSWIDEDataset


class MirrorNUSWIDEDataset(NUSWIDEDataset, BadNetsBaseDataset):
    def __init__(self, dataset_name, data_path, args={}, mirror_args={}):
        NUSWIDEDataset.__init__(self, dataset_name, data_path, args)
        BadNetsBaseDataset.__init__(
            self, self.train_dataset, self.valid_dataset, mirror_args
        )
