#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List

import numpy as np

from secretflow.utils.compressor import STCSparse

COMPRESS_STRATEGY = ("fed_stc", "fed_scr")


def stc_compress(compressor, server_weights, agg_updates, res):
    def _add(matrices_a: List, matrices_b: List):
        results = [np.add(a, b) for a, b in zip(matrices_a, matrices_b)]
        return results

    def _subtract(matrices_a: List, matrices_b: List):
        results = [np.subtract(a, b) for a, b in zip(matrices_a, matrices_b)]
        return results

    if res:
        agg_updates = _add(agg_updates, res)
    sparse_agg_updates = compressor(agg_updates)
    res = _subtract(agg_updates, sparse_agg_updates)
    server_weights = _add(server_weights, sparse_agg_updates)
    return server_weights, sparse_agg_updates, res


def do_compress(
    strategy="batch",
    sparsity=0.0,
    server_weights=None,
    updates=None,
    res=None,
):
    if strategy == "fed_stc":
        compressor = STCSparse(sparse_rate=sparsity)
        return stc_compress(compressor, server_weights, updates, res)

    else:
        return server_weights, updates, res
