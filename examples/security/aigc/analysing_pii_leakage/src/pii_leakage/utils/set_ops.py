# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data


def intersection(a: dict, b: dict):
    """ Gets PII -> count dicts and returns the intersecting dict """
    return {k: v for k, v in a.items() if k in b.keys()}


def difference(a: dict, b: dict):
    """ Gets PII -> count dicts and returns the difference dict """
    return {k: v for k, v in a.items() if k not in b.keys()}


def union(a: dict, b: dict):
    """ Gets PII -> count dicts and returns the union dict """
    return {**a, **b}
