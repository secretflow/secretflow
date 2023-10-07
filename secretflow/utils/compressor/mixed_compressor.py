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
import logging
from typing import Any, List

from secretflow.utils.compressor import SparseCompressor, CompressedData
from secretflow.utils.compressor.base import Compressor
from secretflow.utils.compressor.quantized_compressor import (
    QuantizedCompressor,
    QuantizedCompressedData,
)
from secretflow.utils.compressor.sparse_compressor import SparseCompressedData


class MixedCompressedData(CompressedData):
    """Mixed compressed data.
    The compressed_data is type of Any.
    The compressed_participants is a list, which include several CompressedData associated with mutiple compressors.
    All partitciptants CompressedData's data is None, at where the real data is in self.compressed_data
    """

    def __init__(self, compressed_data, compressed_participants: List[CompressedData]):
        super().__init__(compressed_data)
        self.compressed_participants: list = compressed_participants

    def get_sparse_mask(self):
        return self.compressed_participants[0].get_sparse_mask()


class MixedCompressor(Compressor):
    """A compressor that supports mixed use of quantized and sparse compressors.
    TODO: Support more types of compressor combinations (like 3 compressors or more) @xiaonan.
    """

    def __init__(self, *compressors: Compressor):
        self.comressors = (
            list(compressors) if isinstance(compressors, tuple) else [compressors]
        )
        self._check_input_compressors()

    def _check_input_compressors(self):
        """The user may pass multiple compressors, however, not every combination of compressors canbe used.
        For example:
            [QuantizedCompressor1, QuantizedCompressor2] is not allowed, since quantized compress can be done only once.
        Currently, we only support a Sparse compressor + QuntizedCompressor.
        """
        assert (
            len(self.comressors) == 2
        ), f"Only support 2 compressors (Sparse + Quantized)"
        if isinstance(self.comressors[0], QuantizedCompressor) and isinstance(
            self.comressors[1], SparseCompressor
        ):
            self.comressors.reverse()
            logging.warning(
                "In mixed compressors, sf will first use sparse compress and then use quantized compress."
            )
        if not (
            isinstance(self.comressors[0], SparseCompressor)
            and isinstance(self.comressors[1], QuantizedCompressor)
        ):
            raise RuntimeError(
                f"Only support 2 compressors (Sparse + Quantized), got {type(self.comressors[0]),} and {type(self.comressors[1])}"
            )

    def _compress_one(self, data, sparse_mask=None) -> "MixedCompressedData":
        sparse_compressor = self.comressors[0]
        quantized_compressor = self.comressors[1]
        return self._sparse_quantized_compress(
            data, sparse_compressor, quantized_compressor, sparse_mask
        )

    def _decompress_one(self, data: "MixedCompressedData"):
        sparse_compressor = self.comressors[0]
        quantized_compressor = self.comressors[1]
        return self._sparse_quantized_decompress(
            data, sparse_compressor, quantized_compressor
        )

    def _sparse_quantized_compress(
        self, data, sparse_compressor, quantized_compressor, sparse_mask
    ):
        sparse_data: SparseCompressedData = sparse_compressor.compress(
            data, sparse_mask=sparse_mask
        )
        quantized_data: QuantizedCompressedData = quantized_compressor.compress(
            sparse_data.compressed_data
        )
        ret_data = quantized_data.compressed_data
        sparse_data.compressed_data = None
        quantized_data.compressed_data = None
        return MixedCompressedData(ret_data, [sparse_data, quantized_data])

    def _sparse_quantized_decompress(
        self, data: "MixedCompressedData", sparse_compressor, quantized_compressor
    ):
        compressed_data: Any = data.compressed_data
        quantized_data: QuantizedCompressedData = data.compressed_participants[1]
        quantized_data.compressed_data = compressed_data
        unquantized_data = quantized_compressor.decompress(quantized_data)
        sparse_data: SparseCompressedData = data.compressed_participants[0]
        sparse_data.compressed_data = unquantized_data
        dense_data = sparse_compressor.decompress(sparse_data)
        return dense_data
