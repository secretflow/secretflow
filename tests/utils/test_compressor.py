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

import numpy as np
import numpy.testing as npt

from secretflow_fl.utils.compressor import (
    MixedCompressor,
    QuantizedCompressor,
    QuantizedFP,
    QuantizedKmeans,
    QuantizedLSTM,
    QuantizedZeroPoint,
    RandomSparse,
    SparseCompressor,
    TopkSparse,
)
from secretflow_fl.utils.compressor.mixed_compressor import MixedCompressedData
from secretflow_fl.utils.compressor.quantized_compressor import QuantizedCompressedData
from secretflow_fl.utils.compressor.sparse_compressor import SparseCompressedData


def do_test_quantized_compressor(
    compressor: QuantizedCompressor, data_shape=(10, 10), tolarent=0.1
):
    data = np.random.uniform(low=-5, high=5, size=data_shape)
    compressed_data = compressor.compress(data)
    assert isinstance(compressed_data, QuantizedCompressedData)
    plain_data = compressor.decompress(compressed_data)
    npt.assert_allclose(data, plain_data, atol=tolarent)


def do_tests_sparse_compressor(compressor: SparseCompressor, data_shape=(10, 10)):
    data = np.random.uniform(low=-5, high=5, size=data_shape)
    compressed_data = compressor.compress(data)
    assert isinstance(compressed_data, SparseCompressedData)
    plain_data = compressor.decompress(compressed_data)

    mask = compressed_data.get_sparse_mask()
    compressed_data2 = compressor.compress(data, sparse_mask=mask)
    plain_data2 = compressor.decompress(compressed_data2)
    npt.assert_equal(plain_data, plain_data2)


def do_tests_mix_compressor(compressor: MixedCompressor, data_shape=(10, 10)):
    data = np.random.uniform(low=-5, high=5, size=data_shape)
    compressed_data = compressor.compress(data)
    assert isinstance(compressed_data, MixedCompressedData)
    plain_data = compressor.decompress(compressed_data)
    mask = compressed_data.get_sparse_mask()
    compressed_data2 = compressor.compress(data, sparse_mask=mask)
    plain_data2 = compressor.decompress(compressed_data2)
    npt.assert_equal(plain_data, plain_data2)


def test_topk_sparse_compressor():
    do_tests_sparse_compressor(TopkSparse(0.9))
    do_tests_sparse_compressor(TopkSparse(0.9), data_shape=(10, 10, 10, 10))


def test_random_sparse_compressor():
    do_tests_sparse_compressor(RandomSparse(0.9))
    do_tests_sparse_compressor(RandomSparse(0.9), data_shape=(10, 10, 10, 10))


def test_quantized_lstm_compressor():
    do_test_quantized_compressor(QuantizedLSTM(), tolarent=0.2)
    do_test_quantized_compressor(
        QuantizedLSTM(), data_shape=(10, 10, 10, 10), tolarent=0.2
    )


def test_quantized_zero_point_compressor():
    do_test_quantized_compressor(QuantizedZeroPoint(), tolarent=0.2)
    do_test_quantized_compressor(
        QuantizedZeroPoint(), data_shape=(10, 10, 10, 10), tolarent=0.2
    )


def test_quantized_fp_compressor():
    do_test_quantized_compressor(QuantizedFP(), tolarent=0.2)
    do_test_quantized_compressor(
        QuantizedFP(), data_shape=(10, 10, 10, 10), tolarent=0.2
    )


def test_quantized_kmeans_compressor():
    do_test_quantized_compressor(QuantizedKmeans(), tolarent=4)
    do_test_quantized_compressor(QuantizedKmeans(), data_shape=(10, 10, 10), tolarent=4)


def test_mixed_compressor():
    do_tests_mix_compressor(MixedCompressor(TopkSparse(0.9), QuantizedZeroPoint()))
    do_tests_mix_compressor(
        MixedCompressor(TopkSparse(0.9), QuantizedZeroPoint()), data_shape=(10, 10, 10)
    )
    do_tests_mix_compressor(MixedCompressor(RandomSparse(0.8), QuantizedLSTM()))
    do_tests_mix_compressor(
        MixedCompressor(RandomSparse(0.8), QuantizedLSTM()), data_shape=(10, 10, 10)
    )
