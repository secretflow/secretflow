#!/usr/bin/env python
# coding=utf-8
import numpy as np
import json
import pdb


def asr(preds, truthes, target_class, indexes, hope_indexes):
    # t_indexes = truthes != target_class
    # p_indexes = preds == target_class
    # return np.sum(t_indexes & p_indexes) / np.sum(t_indexes)
    target_indexes = indexes[preds == target_class]
    correct = len(np.intersect1d(target_indexes, hope_indexes))
    return correct / len(hope_indexes)


def load_result(file):
    results = []
    with open(file, "r") as fp:
        for line in fp:
            r = json.loads(line)
            results.append(r)
    return results
