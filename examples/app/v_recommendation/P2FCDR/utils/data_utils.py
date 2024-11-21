# -*- coding: utf-8 -*-
import numpy as np
import torch
from dataset import RecDataset


def load_ratings_dataset(args):
    client_train_datasets = []
    client_valid_datasets = []
    client_test_datasets = []
    for domain in args.domains:
        model = args.method.replace("Fed", "")

        train_dataset = RecDataset(
            args, domain, model, mode="train", load_prep=args.load_prep)
        valid_dataset = RecDataset(
            args, domain, model, mode="valid", load_prep=args.load_prep)
        test_dataset = RecDataset(
            args, domain, model, mode="test", load_prep=args.load_prep)

        client_train_datasets.append(train_dataset)
        client_valid_datasets.append(valid_dataset)
        client_test_datasets.append(test_dataset)
    return client_train_datasets, client_valid_datasets, client_test_datasets
