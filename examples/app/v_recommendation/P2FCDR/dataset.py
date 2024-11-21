# -*- coding: utf-8 -*-
"""Customized dataset.
"""
import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class RecDataset(Dataset):
    """A customized dataset reading and preprocessing data of a certain domain
    from ".txt" files.
    """
    data_dir = "data"
    prep_dir = "prep_data"
    # The number of negative samples to test all methods (including ours)
    num_test_neg = 999
    # num_test_neg = 99
    num_neg = 99

    def __init__(self, args, domain, model="GNN", mode="train",
                 load_prep=True):
        assert (model in ["P2FCDR"])
        assert (mode in ["train", "valid", "test"])

        self.args = args
        self.domain = domain
        self.model = model
        self.mode = mode
        self.dataset_dir = os.path.join(self.data_dir, self.domain + "_"
                                        + "".join([domain[0] for domain
                                                   in self.args.domains]))
        self.user_ids, self.interactions, self.user_items, \
            self.num_users, self.num_items = self.read_data(self.dataset_dir)
        self.prep_interactions = self.preprocess(
            self.interactions, self.dataset_dir, load_prep)

    def read_data(self, dataset_dir):
        with open(os.path.join(dataset_dir, "num_users.txt"),
                  "rt", encoding="utf-8") as infile:
            num_users = int(infile.readline())
        with open(os.path.join(dataset_dir, "num_items.txt"),
                  "rt", encoding="utf-8") as infile:
            num_items = int(infile.readline())

        with open(os.path.join(self.dataset_dir,
                               "%s_data.txt" % self.mode), "rt",
                  encoding="utf-8") as infile:
            user_ids, items = [], []
            user_items = {}
            for line in infile.readlines():
                user, item = line.strip().split("\t")
                user, item = int(user), int(item)
                user_ids.append(user)
                items.append(item)
                if user not in user_items:
                    user_items[user] = set()
                user_items[user].add(item)
        print("Successfully load %s %s data!" % (self.domain, self.mode))

        return user_ids, items, user_items, num_users, num_items

    def preprocess(self, interactions, dataset_dir, load_prep):
        prep_functions = {
                          "P2FCDR": self.preprocess_baselines,
                          }
        if not os.path.exists(os.path.join(dataset_dir, self.prep_dir)):
            os.makedirs(os.path.join(dataset_dir, self.prep_dir))

        self.prep_data_path = os.path.join(
            dataset_dir, self.prep_dir, "%s_%s_data.pkl" % (self.model,
                                                            self.mode))
        if os.path.exists(self.prep_data_path) and load_prep:
            with open(os.path.join(self.prep_data_path), "rb") as infile:
                prep_interactions = pickle.load(infile)
            print("Successfully load preprocessed %s %s data!" %
                  (self.domain, self.mode))
        else:
            prep_interactions = prep_functions[self.model](
                interactions, mode=self.mode)
            with open(self.prep_data_path, "wb") as infile:
                pickle.dump(prep_interactions, infile)
            print("Successfully preprocess %s %s data!" %
                  (self.domain, self.mode))
        return prep_interactions

    @ staticmethod
    def random_neg(left, right, excl):  # [left, right)
        sample = np.random.randint(left, right)
        while sample in excl:
            sample = np.random.randint(left, right)
        return sample


    def preprocess_baselines(self, data, mode="train"):
        prep_data = []
        for user_id, item in zip(self.user_ids, data):
            if mode == "train":
                temp = []
                temp.append(item)
                neg_samples = []
                for _ in range(self.num_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=self.user_items[user_id])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)
                prep_data.append(temp)
            else:
                temp = []
                temp.append(item)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=self.user_items[user_id])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)
                prep_data.append(temp)
        return prep_data

    def __len__(self):
        return len(self.prep_interactions)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        interaction = self.prep_interactions[idx]
        return user_id, interaction

    def __setitem__(self, idx, value):
        """To support shuffle operation.
        """
        self.user_ids[idx] = value[0]
        self.prep_interactions[idx] = value[1]

    def __add__(self, other):
        """To support concatenation operation.
        """
        user_ids, prep_interactions = other
        self.user_ids += user_ids
        self.prep_interactions += prep_interactions
        return self