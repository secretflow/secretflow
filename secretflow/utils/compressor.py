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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
from scipy import sparse

import jax.numpy as jnp
from secretflow.utils.communicate import ForwardData
from secretflow.utils.errors import InvalidArgumentError


class Compressor(ABC):
    """Abstract base class for cross device data compressor"""

    @abstractmethod
    def compress(
        self, data: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[Any, List[Any]]:
        """Compress data before send.

        Args:
            data (Union[np.ndarray, List[np.ndarray]]): data need to compress.

        Returns:
            Union[Any, List[Any]]: compressed data.
        """
        raise NotImplementedError()

    @abstractmethod
    def decompress(
        self, data: Union[Any, List[Any]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Decompress data after receive.

        Args:
            data (Union[Any, List[Any]]): data need to decompress.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: decompressed data.
        """
        raise NotImplementedError()

    @abstractmethod
    def iscompressed(self, data: Union[Any, List[Any]]) -> Union[bool, List[bool]]:
        """Checks whether data or data array has been compressed.

        Args:
            data (Union[Any, List[Any]]): data need to check.

        Returns:
            Union[bool, List[bool]]: True if data is compressed.
        """
        raise NotImplementedError()


class SparseCompressor(Compressor):
    def __init__(self, sparse_rate: float):
        """Initialize

        Args:
            sparse_rate: the percentage of cells are zero.
        """
        assert (
            0 <= sparse_rate <= 1
        ), f'sparse rate should between 0 and 1, but get {sparse_rate}'
        self.sparse_rate = sparse_rate
        self.fuse_sparse_masks = []

    @abstractmethod
    def _compress_one(self, data: np.ndarray) -> sparse.spmatrix:
        """Compress one data to sparse matrix.
        Args:
            data (np.ndarray): data need to compress.

        Returns:
            sparse.spmatrix: compressed sparse matrix.
        """
        raise NotImplementedError()

    # sample random element from original List[np.ndarray]
    def compress(
        self,
        data: Union[ForwardData, np.ndarray, List[np.ndarray]],
    ) -> Union[sparse.spmatrix, List[sparse.spmatrix]]:
        """Compress data to sparse matrix before send.

        Args:
            data (Union[np.ndarray, List[np.ndarray]]): data need to compress.

        Returns:
            Union[sparse.spmatrix, List[sparse.spmatrix]]: compressed data.
        """
        # there is no need for sparsification in evaluate/predict.
        is_list = True

        if isinstance(data, ForwardData):
            hidden = data.hidden
        else:
            hidden = data

        if isinstance(hidden, (np.ndarray, jnp.ndarray)):
            is_list = False
            hidden = [hidden]
        elif not isinstance(hidden, (list, tuple)):
            raise InvalidArgumentError(f'invalid data: {type(hidden)}')
        out = list(map(lambda d: self._compress_one(d), hidden))
        out = out if is_list else out[0]
        if isinstance(data, ForwardData):
            data.hidden = out
        else:
            data = out
        return data

    def decompress(
        self, data: Union[ForwardData, sparse.spmatrix, List[sparse.spmatrix]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Decompress data from sparse matrix to dense after received.

        Args:
            data (Union[sparse.spmatrix, List[sparse.spmatrix]]): data need to decompress.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: decompressed data.
        """
        # there is no need for sparsification in evaluate/predict.
        is_list = True
        if isinstance(data, ForwardData):
            sparse_hidden = data.hidden
        else:
            sparse_hidden = data

        if sparse.issparse(sparse_hidden):
            is_list = False
            sparse_hidden = [sparse_hidden]
        elif not isinstance(sparse_hidden, (list, tuple)):
            raise InvalidArgumentError(f'invalid data: {type(sparse_hidden)}')
        sparse_hidden = list(map(lambda d: d.todense(), sparse_hidden))
        sparse_hidden = sparse_hidden if is_list else sparse_hidden[0]
        if isinstance(data, ForwardData):
            data.hidden = sparse_hidden
        else:
            data = sparse_hidden
        return data

    def iscompressed(
        self, data: Union[sparse.spmatrix, List[sparse.spmatrix]]
    ) -> Union[bool, List[bool]]:
        """Checks whether data or data array has been compressed.

        Args:
            data (Union[sparse.spmatrix, List[sparse.spmatrix]]): data need to check.

        Returns:
            Union[bool, List[bool]]: True if data is compressed.
        """
        is_list = True
        if sparse.issparse(data):
            is_list = False
            data = [data]
        compressed = list(map(lambda d: sparse.issparse(d), data))
        return compressed if is_list else compressed[0]


class RandomSparse(SparseCompressor):
    """Random sparse compressor compress data by randomly set element to zero."""

    def __init__(self, sparse_rate: float):
        super().__init__(sparse_rate)

    def _compress_one(self, data):
        data_shape = data.shape
        data_flat = data.flatten()
        data_len = data_flat.shape[0]
        mask_num = round((1 - self.sparse_rate) * data_len)
        rng = np.random.default_rng()
        mask_index = rng.choice(data_len, mask_num)
        row, col = np.unravel_index(mask_index, data_shape)
        matrix = sparse.coo_matrix(
            (data_flat[mask_index], (row, col)), shape=data_shape
        )
        return matrix.tocsr()


class TopkSparse(SparseCompressor):
    """Topk sparse compressor use topK algorithm to transfer dense matrix into sparse matrix."""

    def __init__(self, sparse_rate: float):
        super().__init__(sparse_rate)

    def _compress_one(self, data):
        data_shape = data.shape
        data_flat = data.flatten()
        data_len = data_flat.shape[0]
        mask_num = round((1 - self.sparse_rate) * data_len)
        mask_index = np.argpartition(np.abs(data), -mask_num, axis=None)[-mask_num:]
        row, col = np.unravel_index(mask_index, data_shape)
        matrix = sparse.coo_matrix(
            (data_flat[mask_index], (row, col)), shape=data_shape
        )
        return matrix.tocsr()


class STCSparse:
    """Stc sparser, sample TopK element from original Weights
    TODO: rewrite in sl compress manner
    """

    def __init__(self, sparse_rate: float):
        self.sparse_rate = sparse_rate
        self.name = 'STC'

    def __call__(
        self,
        weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        compression_weights = []
        mask_arrays = []
        for weight in weights:
            weight_shape = weight.shape
            weight_flat = weight.flatten()
            weight_flat_abs = np.abs(weight_flat)
            weight_len = weight_flat.shape[0]
            mask_num = round(self.sparse_rate * weight_len)
            mask_index = np.sort(np.argsort(weight_flat_abs)[:mask_num])
            mask_array = np.ones(weight_flat.shape)
            if mask_index.shape[0] != 0:
                weight_flat[mask_index] = 0
                mask_array[mask_index] = 0
            if weight_len == mask_num:
                average_value = 0.0
            else:
                average_value = np.sum(np.absolute(weight_flat)) / (
                    weight_len - mask_num
                )
            weight_compress = average_value * np.sign(weight_flat)
            compression_weight = weight_compress.reshape(weight_shape)
            compression_weights.append(compression_weight)
            mask_array = mask_array.reshape(weight_shape)
            mask_arrays.append(mask_array)
        return compression_weights


class SCRSparse:
    """Stc sparser, sample TopK element from original Weights
    TODO: rewrite in sl compress manner
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.name = 'SCR'

    def __call__(
        self,
        weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        compression_weights = []
        mask_arrays = []
        for weight in weights:
            weight_shape = weight.shape
            if len(weight_shape) == 4:
                # CNN layer
                # Keep the 0th dimension
                sum_0 = np.sum(np.absolute(weight), axis=(1, 2, 3))
                sum_0 = sum_0 / np.max(sum_0)
                index_zero_0 = self.get_dimension(sum_0, self.threshold)
                weight[index_zero_0, :, :, :] = 0.0
                # Keep the 1th dimension
                sum_1 = np.sum(np.absolute(weight), axis=(0, 2, 3))
                sum_1 = sum_1 / np.max(sum_1)
                index_zero_1 = self.get_dimension(sum_1, self.threshold)
                weight[:, index_zero_1, :, :] = 0.0
            if len(weight_shape) == 2:
                # Dense layer
                # Keep the 0th dimension
                sum_0 = np.sum(np.absolute(weight), axis=1)
                sum_0 = sum_0 / np.max(sum_0)
                index_zero_0 = self.get_dimension(sum_0, self.threshold)
                weight[index_zero_0, :] = 0.0
                # Keep the 1th dimension
                sum_1 = np.sum(np.absolute(weight), axis=0)
                sum_1 = sum_1 / np.max(sum_1)
                index_zero_1 = self.get_dimension(sum_1, self.threshold)
                weight[:, index_zero_1] = 0.0
            compression_weight = weight
            compression_weights.append(compression_weight)
            mask_array = np.array(compression_weight, dtype=bool)
            mask_arrays.append(mask_array)
        return compression_weights

    def get_dimension(self, index_value, threshold):
        return np.argwhere(index_value <= threshold)


# Sparse matrix encode and decode
def sparse_encode(
    data: List[np.ndarray],
    encode_method: str = 'coo',
) -> List:
    """Encode the sparse matrix

    Args:
        data: sparse matrix to be compressed
        encode_method: compressed method,support ['coo', 'gcxs']
    Returns:
        encoded_datas: Compressed matrix
    """
    # TODO: support more sparse matrix encoding methods
    if data is None:
        return None
    assert encode_method in [
        'coo',
        'gcxs',
    ], f'Get unsupport sparse encoding method: {encode_method}, '
    encoded_datas = []
    import sparse as sp

    for datum in data:
        if encode_method == 'coo':
            encoded_data = sp.COO(datum)
        else:
            encoded_data = sp.GCXS(datum)
        encoded_datas.append(encoded_data)
    return encoded_datas


def sparse_decode(data: List) -> List[np.ndarray]:
    """Decode the compressed sparse matrix

    Args:
        data: compressed matrix to be decoded
    Returns:
        decoded_datas: Decoded matrix
    """
    import sparse as sp

    if data is None:
        return None
    assert isinstance(
        data[0], (sp._coo.core.COO, sp._compressed.compressed.GCXS)
    ), 'Sparse encoding method not supporterd, Only COO GCXS supported'
    decode_datas = []
    for datum in data:
        decode_datum = datum.todense()
        decode_datas.append(decode_datum)
    return decode_datas


@dataclass
class QuantizedData:
    data: Any = None
    q1: int = None
    q2: int = None
    origin_type: Any = None


class QuantizedCompressor(Compressor):
    """Abstract base class for quantized compressor"""

    def __init__(self, quant_bits: int = 8):
        """Initialize

        Args:
            quant_bits: the compressed bits length.
        """
        super().__init__()
        self.quant_bits = quant_bits
        self.np_type = self._infer_np_type(quant_bits)

    def compress(
        self,
        data: Union[ForwardData, np.ndarray, List[np.ndarray]],
    ) -> Union[Any, List[Any]]:
        is_list = True

        if isinstance(data, ForwardData):
            hidden = data.hidden
        else:
            hidden = data

        if isinstance(hidden, (np.ndarray, jnp.ndarray)):
            is_list = False
            hidden = [hidden]
        elif not isinstance(hidden, (list, tuple)):
            raise InvalidArgumentError(f'invalid data: {type(hidden)}')
        out = list(map(lambda x: self._compress_one(x), hidden))
        out = out if is_list else out[0]
        if isinstance(data, ForwardData):
            data.hidden = out
        else:
            data = out
        return data

    def decompress(
        self,
        data: Union[ForwardData, np.ndarray, List[np.ndarray], List[QuantizedData]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        is_list = True
        if isinstance(data, ForwardData):
            quantized_hidden = data.hidden
        else:
            quantized_hidden = data

        if isinstance(quantized_hidden, (np.ndarray, jnp.ndarray, QuantizedData)):
            is_list = False
            quantized_hidden = [quantized_hidden]

        quantized_hidden = list(
            map(lambda x: self._decompress_one(x), quantized_hidden)
        )
        quantized_hidden = quantized_hidden if is_list else quantized_hidden[0]

        if isinstance(data, ForwardData):
            data.hidden = quantized_hidden
        else:
            data = quantized_hidden

        return data

    def iscompressed(self, data: Union[Any, List[Any]]) -> Union[bool, List[bool]]:
        if not isinstance(data, list):
            return isinstance(data, QuantizedData)
        is_compressed = list(map(lambda x: isinstance(x, QuantizedData), data))
        return is_compressed

    def _infer_np_type(self, quant_bits):
        if self.quant_bits <= 0 or self.quant_bits > 64:
            logging.error(
                f"The quantized bits len must be between 0 and 64, got {quant_bits}"
            )
            raise RuntimeError(
                f"The quantized bits len must be between 0 and 64, got {quant_bits}"
            )
        recommend_bits = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
        if quant_bits in recommend_bits:
            return recommend_bits[quant_bits]
        else:
            logging.warning(
                f"It is recommended to use 8/16/32/64 as the compression bits (got {self.quant_bits}),"
                f"That will loss accuracy but not save communication traffic."
            )
            for i in range(64):
                if (quant_bits + i) in recommend_bits:
                    return recommend_bits[(quant_bits + i)]
        raise RuntimeError(f"Unknown input quant_bits {quant_bits}")

    @abstractmethod
    def _compress_one(self, data: np.ndarray) -> QuantizedData:
        raise NotImplementedError()

    @abstractmethod
    def _decompress_one(self, data: QuantizedData) -> np.ndarray:
        raise NotImplementedError()


class QuantizedZeroPoint(QuantizedCompressor):
    """Quantized compressor with strengthen the QuantizedLSTM with replacing 32-bit RQM to 8-bit zero point.
    The tests show that the QuantizedZeroPoint compressor has higner accuracy than QuantizedLSTM.
    Reference paper 2017 "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference".

    Link: https://arxiv.org/abs/1712.05877
    """

    def __init__(self, quant_bits: int = 8):
        super().__init__(quant_bits)

    def _compress_one(self, data: np.ndarray) -> QuantizedData:
        _max = np.max(data)
        _min = np.min(data)
        qmin = -(int)(1 << (self.quant_bits - 1))
        qmax = (int)((1 << (self.quant_bits - 1)) - 1)

        scale = (_max - _min) / (qmax - qmin)
        initial_zero_point = qmin - _min / scale
        nudged_zero_point = initial_zero_point
        if initial_zero_point < qmin:
            nudged_zero_point = qmin
        elif initial_zero_point > qmax:
            nudged_zero_point = qmax
        else:
            nudged_zero_point = int(initial_zero_point)
        transformed_val = data / scale + nudged_zero_point
        clamped_val = np.clip(transformed_val, qmin, qmax)
        quantized = np.round(clamped_val)
        return QuantizedData(
            quantized.astype(self.np_type), scale, nudged_zero_point, data.dtype
        )

    def _decompress_one(self, data: QuantizedData) -> np.ndarray:
        return (data.data.astype(data.origin_type) - float(data.q2)) * data.q1


class QuantizedLSTM(QuantizedCompressor):
    """Quantized compressor with LSTM, a basic algorithm which replace float with int.

    Reference paper 2016 "On the efficient representation and execution of deep acoustic models".

    Link: https://arxiv.org/abs/1607.04683
    """

    def __init__(self, quant_bits: int = 8):
        super().__init__(quant_bits)

    def _compress_one(self, data: np.ndarray) -> QuantizedData:
        _max = np.max(data)
        _min = np.min(data)
        q_scale = int((1 << self.quant_bits) - 1)
        q_shift = int(1 << (self.quant_bits - 1))

        q = q_scale / (_max - _min)
        rqm = int(round(q * _min) + q_shift)
        quantized = np.round(q * data) - rqm
        return QuantizedData(quantized.astype(self.np_type), q, rqm, data.dtype)

    def _decompress_one(self, data: QuantizedData) -> np.ndarray:
        return (data.data.astype(data.origin_type) + float(data.q2)) / float(data.q1)


class QuantizedFP(QuantizedCompressor):
    """Quantized compressor with low-bit floating points, fp16/32/64 will be change directly in numpy format, while fp8(M4E3) will be stored as int8 object.

    Reference paper "FP8 FORMATS FOR DEEP LEARNING".

    Link: https://arxiv.org/pdf/2209.05433.pdf
    """

    def __init__(self, quant_bits: int = 8):
        super().__init__(quant_bits)
        if quant_bits not in [8, 16, 32, 64]:
            raise RuntimeError(
                f"The quantized bits for QuantizedFP must in 8/16/32/64, got {quant_bits}"
            )

    def _compress_one(self, data: np.ndarray) -> QuantizedData:
        if self.quant_bits > 8:
            # fp 16/32/64
            return QuantizedData(
                data.astype(getattr(np, f'float{self.quant_bits}')),
                None,
                None,
                data.dtype,
            )
        else:
            # fp8(M4E3) with a scale factor, store as np.int8.
            q_sign = np.sign(data)

            out = np.abs(data)
            scale = 448 / np.max(out)  # A fp8 value can represent to [-448, 448]
            out = out * scale
            mant, exp = np.frexp(out)
            q_mant = np.round(mant * 8)
            q_exp = np.where(exp > -6, exp + 6, 0)

            quantized = q_sign * (q_exp * 8 + q_mant)
            return QuantizedData(
                quantized.astype(self.np_type), scale, None, data.dtype
            )

    def _decompress_one(self, data: QuantizedData) -> np.ndarray:
        if self.quant_bits != 8:
            return data.data.astype(data.origin_type)
        else:
            # decompose fp8(stored in int8) to default fp
            quantized = data.data
            sign = np.sign(quantized)
            abs_quantized = np.abs(quantized)
            exp = (abs_quantized // 8) - 6
            mant = (abs_quantized % 8).astype(data.origin_type) / 8
            ori_data = sign * np.ldexp(mant, exp) / data.q1
            return ori_data
