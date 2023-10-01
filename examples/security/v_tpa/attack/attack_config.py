#!/usr/bin/env python
# coding=utf-8
BADNETS_ARGS = {
    'train_poison_ratio': 0.01,
    'valid_poison_ratio': 0.01,
    'train_known_target_num': 15,
    'valid_known_target_num': 15,
    'target_class': None,
}

POISONING_ARGS = {
    'train_poisoning_indexes': None,
    'valid_poisoning_indexes': None,
    'train_target_indexes': None,
    'valid_target_indexes': None,
    'gamma': 20.0,
    'blurred': False,
}
