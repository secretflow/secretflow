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
from abc import abstractmethod

from secretflow.utils.compressor.base import Compressor, CompressedData
from scipy import sparse
import numpy as np
from typing import List


class SparseCompressedData(CompressedData):
    """SparseCompressed data, inside data can be replaced by scipy.sparse.coo_matrix.
    However, for MixedCompressor, we need to combine these parameters with other compressors,
    so we do not construct coo_matrix at the beginning.
    """

    def __init__(self, compressed_data, sparse_rate: float, row, col, shape):
        super().__init__(compressed_data)
        self.sparse_rate = sparse_rate
        self.row = row
        self.col = col
        self.shape = shape

    def to_csr(self):
        return sparse.coo_matrix(
            (self.compressed_data, (self.row, self.col)), shape=self.shape
        ).tocsr()

    def get_sparse_mask(self):
        """We can simply use self.to_csr() != 0 to get a sparse_mask in sparse compressor. However, when use mixed
        compressor, the stored data is processed by quantized compressors or other, which may lead to 0 in data and
        will be droped after self.to_csr() != 0 (0 should not be droped since after unquantized, it may be any
        value.) To prevent this situation, we construct an all 1 matrix, to ensure all candidates values are not droped.
        """
        ones = np.ones((len(self.row),), dtype=bool)
        mask_matix = sparse.csr_matrix((ones, (self.row, self.col)), shape=self.shape)
        return mask_matix


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

    def _decompress_one(self, data: SparseCompressedData):
        return data.to_csr().todense()

    def _compress_one(self, data, sparse_mask=None) -> SparseCompressedData:
        if sparse_mask is not None:
            d = sparse_mask.multiply(data)
            return SparseCompressedData(d.data, self.sparse_rate, d.row, d.col, d.shape)
        else:
            return self._do_compress_one(data)

    @abstractmethod
    def _do_compress_one(self, data):
        raise NotImplementedError()


class RandomSparse(SparseCompressor):
    """Random sparse compressor compress data by randomly set element to zero."""

    def __init__(self, sparse_rate: float):
        super().__init__(sparse_rate)

    def _do_compress_one(self, data) -> SparseCompressedData:
        data_shape = data.shape
        data_flat = data.flatten()
        data_len = data_flat.shape[0]
        mask_num = round((1 - self.sparse_rate) * data_len)
        rng = np.random.default_rng()
        mask_index = rng.choice(data_len, mask_num, replace=False)
        row, col = np.unravel_index(mask_index, data_shape)
        target_data = data_flat[mask_index]
        return SparseCompressedData(target_data, self.sparse_rate, row, col, data_shape)


class TopkSparse(SparseCompressor):
    """Topk sparse compressor use topK algorithm to transfer dense matrix into sparse matrix."""

    def __init__(self, sparse_rate: float):
        super().__init__(sparse_rate)

    def _do_compress_one(self, data, sparse_mask=None):
        data_shape = data.shape
        data_flat = data.flatten()
        data_len = data_flat.shape[0]
        mask_num = round((1 - self.sparse_rate) * data_len)
        mask_index = np.argpartition(np.abs(data), -mask_num, axis=None)[-mask_num:]
        row, col = np.unravel_index(mask_index, data_shape)
        target_data = data_flat[mask_index]
        return SparseCompressedData(target_data, self.sparse_rate, row, col, data_shape)


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
