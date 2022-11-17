import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse
from typing import Union, List, Any


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
