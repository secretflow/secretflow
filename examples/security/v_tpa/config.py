#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
