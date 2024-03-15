#!/usr/bin/env python
# coding=utf-8
from custom_base.gradreplace_sl_base import GRADReplace_PYUSLTFModel
from custom_base.index_sl_base import IndexSLBaseTFModel
from custom_base.replay_sl_base import Replay_PYUSLTFModel

__all__ = [
    "IndexSLBaseTFModel",
    "GRADReplace_PYUSLTFModel",
    "Replay_PYUSLTFModel",
]


METHODS = ["replay", "grad_replacement"]
# AGGREGATIONS = ['average', 'naive_sum', 'sum', 'concatenate']
AGGREGATIONS = ["naive_sum"]
TIMES = 10
PARTY_NUM = 2
PARTIES = [
    "alice",
    "bob",
    "charles",
    "david",
    "ella",
    "filler",
    "ganna",
    "hellen",
    "idiom",
]
