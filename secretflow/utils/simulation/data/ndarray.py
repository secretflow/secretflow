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

from typing import Callable, Dict, List, Union

import numpy as np

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.device import PYU
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data._utils import cal_indexes


def create_ndarray(
    source: Union[str, np.ndarray, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    axis: int = 0,
    shuffle: bool = False,
    random_state: int = None,
    allow_pickle: bool = False,
    is_torch: bool = False,
) -> FedNdarray:
    """Create a federated ndarray from a single data source.

    Args:
        source: the dataset source, shall be a file path or numpy.ndarray or
            callable (shall returns a pandas.DataFrame).
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        axis: optional, the value is 0 or 1. 0 means split by row returning a
            horizontal partitioning federated DataFrame. 1 means split by
            column returning a vertical partitioning federated DataFrame.
        shuffle: optional, if suffule the dataset before split.
        random_state: optional, the random state for shuffle.
        allow_pickle: the np.load argument when source is a  file path.

    Returns:
        a FedNdrray.

    Examples
    --------
    >>> arr = np.array([[1, 2, 3, 4], [11, 12, 13, 14]])

    >>> # Create a horizontal partitioned FedNdarray evenly.
    >>> h_arr = created_ndarray(arr, [alice, bob], axis=0)

    >>> # Create a vertical partitioned FedNdarray.
    >>> v_arr = created_ndarray(arr, {alice: 0.3, bob: 0.7}, axis=1)

    >>> # Create a horizontal partitioned FedNdarray evenly.
    >>> h_arr = created_ndarray(arr, {alice: (0, 1), bob: (1, 4)})
    """

    assert parts, 'Parts should not be none or empty!'

    if isinstance(source, str):
        arr = np.load(source, allow_pickle=allow_pickle)
    elif isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, Callable):
        arr = source()
        assert isinstance(
            arr, np.ndarray
        ), f'Callable source must return a ndarray but got {type(arr)}'
    else:
        raise InvalidArgumentError(
            f'Unknown source type, expect a file or ndarray or callable but got {type(source)}'
        )

    if is_torch:
        arr = arr[:, np.newaxis, :, :]

    if shuffle:
        arr = np.random.default_rng(random_state).shuffle(arr)

    total_num = arr.shape[0] if axis == 0 else arr.shape[1]
    assert total_num >= len(
        parts
    ), f'Total samples/columns {total_num} is less than parts number {len(parts)}.'

    indexes = cal_indexes(parts=parts, total_num=total_num)

    if axis == 0:
        return FedNdarray(
            partitions={
                device: device(lambda df: df[index[0] : index[1]])(arr)
                for device, index in indexes.items()
            },
            partition_way=PartitionWay.HORIZONTAL,
        )
    else:
        return FedNdarray(
            partitions={
                device: device(lambda df: df[:, index[0] : index[1]])(arr)
                for device, index in indexes.items()
            },
            partition_way=PartitionWay.VERTICAL,
        )
