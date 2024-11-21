# -*- coding: utf-8 -*-
"""Customized dataloader.
"""
import random
import numpy as np
from torch.utils.data import DataLoader


class RecDataloader(DataLoader):
    """A customized dataloader class iterating over the customized dataset.
    """

    def __init__(self, dataset, batch_size=128, shuffle=True):
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.mode = dataset.mode
        self.batch_size = batch_size
        self.shuffle = True

        if shuffle:
            random.shuffle(self.dataset)

        if len(dataset) % batch_size == 0:
            self.num_batch = len(dataset) // batch_size
        else:
            self.num_batch = len(dataset) // batch_size + 1
            if self.mode == "train":
                # Concatenate
                self.dataset += self.dataset[: batch_size -
                                             len(dataset) % batch_size]

    def __iter__(self):
        for batch_idx in range(self.num_batch):
            start_idx = batch_idx * self.batch_size
            batch_user_ids, batch_interactions = self.dataset[start_idx:
                                                              start_idx +
                                                              self.batch_size]
            batch_interactions = list(zip(*batch_interactions))
            yield np.array(batch_user_ids), tuple(np.array(x) for x
                                                  in batch_interactions)