#!/usr/bin/env python
# coding=utf-8
import random

from torch.utils.data import Sampler


class ShuffleSampler(Sampler):
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.n = len(self.data_source)
        self.indices = list(range(self.n))
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed)
        random.shuffle(self.indices)

        return iter(self.indices)

    def __len__(self):
        return self.n
