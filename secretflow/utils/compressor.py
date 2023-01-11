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

from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
import sparse as sp
from scipy import sparse


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
        data: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[sparse.spmatrix, List[sparse.spmatrix]]:
        """Compress data to sparse matrix before send.

        Args:
            data (Union[np.ndarray, List[np.ndarray]]): data need to compress.

        Returns:
            Union[sparse.spmatrix, List[sparse.spmatrix]]: compressed data.
        """
        # there is no need for sparsification in evaluate/predict.
        is_list = True
        if isinstance(data, np.ndarray):
            is_list = False
            data = [data]
        elif not isinstance(data, (list, tuple)):
            assert False, f'invalid data: {type(data)}'
        out = list(map(lambda d: self._compress_one(d), data))
        return out if is_list else out[0]

    def decompress(
        self, data: Union[sparse.spmatrix, List[sparse.spmatrix]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Decompress data from sparse matrix to dense after received.

        Args:
            data (Union[sparse.spmatrix, List[sparse.spmatrix]]): data need to decompress.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: decompressed data.
        """
        # there is no need for sparsification in evaluate/predict.
        is_list = True
        if sparse.issparse(data):
            is_list = False
            data = [data]
        elif not isinstance(data, (list, tuple)):
            assert False, f'invalid data: {type(data)}'
        data = list(map(lambda d: d.todense(), data))
        return data if is_list else data[0]

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
