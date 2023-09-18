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

import jax.numpy as jnp
import numpy as np

from secretflow.utils.communicate import ForwardData
from secretflow.utils.errors import InvalidArgumentError


class CompressedData:
    """Data after compressed"""

    def __init__(self, compressed_data):
        self.compressed_data = compressed_data


class Compressor(ABC):
    """Abstract base class for cross device data compressor"""

    def compress(
        self, data: Union[np.ndarray, List[np.ndarray]], **kwargs
    ) -> CompressedData:
        """Compress data before send.

        Args:
            data (Union[np.ndarray, List[np.ndarray]]): data need to compress.

        Returns:
            Union[Any, List[Any]]: compressed data.
        """
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
        out = list(map(lambda d: self._compress_one(d, **kwargs), hidden))
        out = out if is_list else out[0]
        if isinstance(data, ForwardData):
            data.hidden = out
        else:
            data = out
        return data

    def decompress(
        self, data: Union[Any, List[Any]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Decompress data after receive.

        Args:
            data (Union[Any, List[Any]]): data need to decompress.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: decompressed data.
        """
        is_list = True
        if isinstance(data, ForwardData):
            hidden = data.hidden
        else:
            hidden = data

        if not isinstance(hidden, list):
            is_list = False
            hidden = [hidden]
        elif not isinstance(hidden, (list, tuple)):
            raise InvalidArgumentError(f'invalid data: {type(hidden)}')
        hidden = list(map(self._decompress_one, hidden))
        hidden = hidden if is_list else hidden[0]
        if isinstance(data, ForwardData):
            data.hidden = hidden
        else:
            data = hidden
        return data

    def iscompressed(self, data: Union[Any, List[Any]]) -> Union[bool, List[bool]]:
        """Checks whether data or data array has been compressed.

        Args:
            data (Union[Any, List[Any]]): data need to check.

        Returns:
            Union[bool, List[bool]]: True if data is compressed.
        """
        if not isinstance(data, list):
            return isinstance(data, CompressedData)
        is_compressed = list(map(lambda x: isinstance(x, CompressedData), data))
        return is_compressed

    @abstractmethod
    def _compress_one(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _decompress_one(self, data):
        raise NotImplementedError()
