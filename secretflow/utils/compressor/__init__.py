# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Union

from .base import CompressedData, Compressor
from .mixed_compressor import MixedCompressor
from .quantized_compressor import (
    QuantizedCompressor,
    QuantizedFP,
    QuantizedKmeans,
    QuantizedLSTM,
    QuantizedZeroPoint,
)
from .sparse_compressor import (
    RandomSparse,
    SCRSparse,
    SparseCompressor,
    STCSparse,
    TopkSparse,
    sparse_decode,
    sparse_encode,
)


def get(name: str, params: Union[Dict, List]):
    simple_compressors = {
        "random_sparse": RandomSparse,
        "topk_sparse": TopkSparse,
        "stc_sparse": STCSparse,
        "scr_sparse": SCRSparse,
        "quantized_fp": QuantizedFP,
        "quantized_kmeans": QuantizedKmeans,
        "quantized_lstm": QuantizedLSTM,
        "quantized_zeropoint": QuantizedZeroPoint,
    }

    mixed_compressors = {
        "mixed_compressor": MixedCompressor,
    }

    if name in simple_compressors:
        return simple_compressors[name](**params)

    if name in mixed_compressors:
        assert isinstance(params, List)
        compressors = []
        for p in params:
            assert isinstance(p, Dict) and "name" in p
            compressors.append(get(p["name"], p.get("params")))

        return mixed_compressors[name](*compressors)

    raise ValueError(f"Compressor '{name}' not exists.")


__all__ = [
    "get",
    "Compressor",
    "CompressedData",
    "sparse_decode",
    "sparse_encode",
    "SparseCompressor",
    "RandomSparse",
    "TopkSparse",
    "STCSparse",
    "SCRSparse",
    "QuantizedCompressor",
    "QuantizedFP",
    "QuantizedKmeans",
    "QuantizedLSTM",
    "QuantizedZeroPoint",
    "MixedCompressor",
]
