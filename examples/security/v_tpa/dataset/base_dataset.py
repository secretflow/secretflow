#!/usr/bin/env python
# coding=utf-8
class BaseDataset:
    def __init__(self, dataset_name, data_path):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train_dataset = None
        self.valid_dataset = None
        self.train_indexes = None
        self.valid_indexes = None
        self.label_set = None

        self.train_pdatasets = None
        self.train_adataset = None
        self.valid_pdatasets = None
        self.valid_adataset = None

    def get_label_set(self):
        return self.label_set

    def load_dataset(self):
        raise "load_dataset() should be implemented in derived class!!!"

    def _split_data(self, dataset, party_num):
        raise "load_dataset() should be implemented in derived class!!!"

    def split_train(self, party_num=2, channel_first=True):
        if self.train_pdatasets is None:
            self.train_pdatasets, self.train_adataset = self._split_data(
                self.train_dataset, party_num, channel_first
            )
        return self.train_pdatasets, self.train_adataset

    def split_valid(self, party_num=2, channel_first=True):
        if self.valid_pdatasets is None:
            self.valid_pdatasets, self.valid_adataset = self._split_data(
                self.valid_dataset, party_num, channel_first
            )
        return self.valid_pdatasets, self.valid_adataset
