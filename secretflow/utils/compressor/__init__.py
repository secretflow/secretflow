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
