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
from secretflow.utils.simulation.data._utils import (
    SPLIT_METHOD,
    dirichlet_partition,
    iid_partition,
    label_skew_partition,
)


def create_ndarray(
    source: Union[str, np.ndarray, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    axis: int = 0,
    shuffle: bool = False,
    random_state: int = None,
    allow_pickle: bool = False,
    is_torch: bool = False,
    sample_method: SPLIT_METHOD = SPLIT_METHOD.IID,
    target: np.ndarray = None,
    is_label: bool = False,
    **kwargs,
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
        sample_method: the sample method to produce index,default 'iid' method.
        target: optional, the target label ndarray.
        kwargs: optional, will accept params for other split method, such as dirichlet_partition and laebl_skew.

        is_torch: torch mode need a new axis.
        is_label: if the input data is label and axis = 1, then we do not need to split.
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

    >>> # Create a horizontal partitioned FedNdarray with DIRICHLET partition method.
    >>> h_arr = created_ndarray(arr, [alice,bob], axis=0, split_method=SPLIT_METHOD.DIRICHLET, num_classes=2, alpha=10000)

    >>> # Create a horizontal partitioned FedNdarray with LABEL_SKEW partition method.
    >>> h_arr = created_ndarray(arr, [alice,bob], axis=0, split_method=SPLIT_METHOD.LABEL_SKEW, label_column='f3', skew_ratio=0.5)
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
    else:
        random_state = np.random.randint(0, 100000)
    if is_label and axis != 0:
        device_list = list(parts.keys()) if isinstance(parts, Dict) else list(parts)
        return FedNdarray(
            partitions={device: device(lambda df: df)(arr) for device in device_list},
            partition_way=PartitionWay.VERTICAL,
        )

    total_num = arr.shape[axis]
    assert total_num >= len(
        parts
    ), f'Total samples/columns {total_num} is less than parts number {len(parts)}.'
    if sample_method == SPLIT_METHOD.IID:
        indexes = iid_partition(
            parts=parts,
            total_num=total_num,
            shuffle=shuffle,
            random_seed=random_state,
        )
    elif sample_method == SPLIT_METHOD.DIRICHLET and axis == 0:
        num_classes = kwargs.pop("num_classes", 0)
        alpha = kwargs.pop("alpha", 10000)
        assert num_classes > 0, "dirichlet partition must supply num_classes"
        assert target is not None, "dirichlet partition must supply target"
        indexes = dirichlet_partition(
            parts=parts,
            targets=target,
            num_classes=num_classes,
            alpha=alpha,
            random_seed=random_state,
        )
    elif sample_method == SPLIT_METHOD.LABEL_SCREW and axis == 0:
        num_classes = kwargs.pop('num_classes', 0)
        max_class_nums = kwargs.pop('max_class_nums', num_classes)
        assert num_classes > 0, "dirichlet partition must supply num_classes"
        assert target is not None, "dirichlet partition must supply target"
        indexes = label_skew_partition(
            parts=parts,
            targets=target,
            num_classes=num_classes,
            max_class_nums=max_class_nums,
            random_seed=random_state,
        )
    else:
        raise Exception(f'Unsupported sample method: {sample_method}.  axis = {axis}')
    if axis == 0:
        return FedNdarray(
            partitions={
                device: device(lambda ndarray, index: ndarray[index, ...])(arr, index)
                for device, index in indexes.items()
            },
            partition_way=PartitionWay.HORIZONTAL,
        )
    else:

        def get_slice(arr, index, is_label, axis):
            if is_label:
                return arr
            else:
                new_arr = np.swapaxes(arr, 0, axis)
                slice_arr = new_arr[index, ...]
                slice_arr = np.swapaxes(slice_arr, 0, axis)
                return slice_arr

        return FedNdarray(
            partitions={
                device: device(get_slice)(arr, index, is_label, axis)
                for device, index in indexes.items()
            },
            partition_way=PartitionWay.VERTICAL,
        )
