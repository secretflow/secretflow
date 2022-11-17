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

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union, Optional

from enum import Enum, unique
import numpy as np
import jax.numpy as jnp

from sklearn.model_selection import train_test_split as _train_test_split

from secretflow.data.io import util as io_util
from secretflow.device import PYU, PYUObject, reveal, SPU, to
from secretflow.utils.errors import InvalidArgumentError

from .math_utils import (
    sum_of_difference_squares,
    sum_of_difference_abs,
    sum_of_difference_ratio_abs,
    mean_of_difference_squares,
    mean_of_difference_abs,
    mean_of_difference_ratio_abs,
)

# 下面的函数是同时支持水平和垂直的。
__ndarray = "__ndarray_type__"


@unique
class PartitionWay(Enum):
    """The partitioning.
    HORIZONTAL: horizontal partitioning.
    VERATICAL: vertical partitioning.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class FedNdarray:
    """Horizontal or vertical partitioned Ndarray.

    Attributes:
        partitions (Dict[PYU, PYUObject]): List of references to
            local numpy.ndarray that makes up federated ndarray.
    """

    partitions: Dict[PYU, PYUObject]
    partition_way: PartitionWay

    @reveal
    def partition_shape(self):
        """Get ndarray shapes of all partitions."""
        return {
            device: device(lambda partition: partition.shape)(partition)
            for device, partition in self.partitions.items()
        }

    @reveal
    def partition_size(self):
        """Get ndarray sizes of all partitions."""
        return {
            device: device(lambda partition: partition.size)(partition)
            for device, partition in self.partitions.items()
        }

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of united ndarray."""
        shapes = self.partition_shape()

        if len(shapes) == 0:
            return (0, 0)

        # no check shapes. assume arrays are aligned by split axis and has same dimension.
        first_shape = list(shapes.values())[0]
        assert len(first_shape) <= 2, "only support get shape on 1/2-D array"

        if len(first_shape) == 1:
            # 1D-array
            assert (
                len(shapes) == 1
            ), "can not get shape on 1D-array with multiple partitions"
            return first_shape

        if self.partition_way == PartitionWay.VERTICAL:
            rows = first_shape[0]
            cols = sum([shapes[d][1] for d in shapes])
        else:
            rows = sum([shapes[d][0] for d in shapes])
            cols = first_shape[1]

        return (rows, cols)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """Cast to a specified type.

        All args are same with :py:meth:`numpy.ndarray.astype`.
        """
        return FedNdarray(
            partitions={
                device: device(
                    lambda a, dtype, order, casting, subok, copy: a.astype(
                        dtype, order=order, casting=casting, subok=subok, copy=copy
                    )
                )(partition, dtype, order, casting, subok, copy)
                for device, partition in self.partitions.items()
            },
            partition_way=self.partition_way,
        )

    def __getitem__(self, item) -> "FedNdarray":
        return FedNdarray(
            partitions={
                pyu: pyu(np.ndarray.__getitem__)(self.partitions[pyu], item)
                for pyu in self.partitions
            },
            partition_way=self.partition_way,
        )

    def __sub__(self, other: "FedNdarray"):
        assert check_same_partition_shapes(
            self, other
        ), "- operator is only supported for same partition shape FedNdarrays"
        new_partitions = {
            device: device(lambda x, y: x - y)(
                self.partitions[device], other.partitions[device]
            )
            for device in self.partitions.keys()
        }
        return FedNdarray(new_partitions, self.partition_way)

    def __add__(self, other: "FedNdarray"):
        assert check_same_partition_shapes(
            self, other
        ), "+ operator is only supported for same partition shape FedNdarrays"
        new_partitions = {
            device: device(lambda x, y: x + y)(
                self.partitions[device], other.partitions[device]
            )
            for device in self.partitions.keys()
        }
        return FedNdarray(new_partitions, self.partition_way)


def subtract(y1: FedNdarray, y2: FedNdarray, spu_device: Optional[SPU] = None):
    """
    subtraction of two FedNdarray object

     Args:
        y1: FedNdarray
        y2: FedNdarray
        spu_device: Optional SPU device

    Returns:
        result of subtraction

    as long as they have the same shape, the result is computable.
    They may have different partition shapes.
    """

    def spu_subtract_dispatcher(axis):
        def spu_subtract(obj_list1: List[np.ndarray], obj_list2: List[np.ndarray]):
            return jnp.concatenate(
                [obj_list1[i] - obj_list2[i] for i in range(len(obj_list1))], axis
            )

        return spu_subtract

    return binary_op(
        spu_subtract_dispatcher(get_concat_axis(y1)), jnp.subtract, y1, y2, spu_device
    )


def load(
    sources: Dict[PYU, Union[str, Callable[[], np.ndarray], PYUObject]],
    partition_way: PartitionWay = PartitionWay.VERTICAL,
    allow_pickle=False,
    encoding="ASCII",
) -> FedNdarray:
    """Load FedNdarray from data source.

    .. warning:: Loading files that contain object arrays uses the ``pickle``
                 module, which is not secure against erroneous or maliciously
                 constructed data. Consider passing ``allow_pickle=False`` to
                 load data that is known not to contain object arrays for the
                 safer handling of untrusted sources.

    Args:
        sources: Data source in each partition. Shall be one of the followings.
            1) Loaded numpy.ndarray.
            2) Local filepath which should be `.npy` or `.npz` file.
            3) Callable function that return numpy.ndarray.
        allow_pickle: Allow loading pickled object arrays stored in npy files.
        encoding: What encoding to use when reading Python 2 strings.

    Raises:
        TypeError: illegal source。

    Returns:
        Returns a FedNdarray if source is pyu object or .npy. Or return a dict
        {key: FedNdarray} if source is .npz.

    Examples:
        >>> fed_arr = load({'alice': 'example/alice.csv', 'bob': 'example/alice.csv'})
    """

    def _load(content) -> Tuple[List, List]:
        if isinstance(content, str):
            data = np.load(
                io_util.open(content), allow_pickle=allow_pickle, encoding=encoding
            )
        elif isinstance(content, Callable):
            data = content()
        else:
            raise TypeError(f"Unsupported source with {type(content)}.")
        assert isinstance(data, np.ndarray) or isinstance(data, np.lib.npyio.NpzFile)
        if isinstance(data, np.lib.npyio.NpzFile):
            files = data.files
            data_list = []
            for file in files:
                data_list.append(data[file])
            return files, data_list
        else:
            return [__ndarray], [data]

    def _get_item(file_idx, data):
        return data[file_idx]

    file_list = []
    data_dict = {}
    pyu_parts = {}

    for device, content in sources.items():
        if isinstance(content, PYUObject) and content.device != device:
            raise InvalidArgumentError("Device of source differs with its key.")
        if not isinstance(content, PYUObject):
            files, datas = device(_load)(content)
            file_list.append(reveal(files))
            data_dict[device] = datas
        else:
            pyu_parts[device] = content
    # 处理pyu object
    if pyu_parts:
        return FedNdarray(partitions=pyu_parts, partition_way=partition_way)

    # 检查各方的数据是否一致
    file_list_lens = set([len(file) for file in file_list])
    if len(file_list_lens) != 1:
        raise Exception(
            f"All parties should have same structure,but got file_list = {file_list}"
        )

    file_names = file_list[0]
    result = {}
    for idx, m in enumerate(file_names):
        parts = {}
        for device, data in data_dict.items():
            parts[device] = device(_get_item)(idx, data)
        if m == __ndarray and len(file_names) == 1:
            return FedNdarray(partitions=parts, partition_way=partition_way)
        result[m] = FedNdarray(partitions=parts, partition_way=partition_way)
    return result


def train_test_split(
    data: FedNdarray, ratio: float, random_state: int = None, shuffle=True
) -> Tuple[FedNdarray, FedNdarray]:
    """Split data into train and test dataset.

    Args:
        data: Data to split.
        ratio: Train dataset ratio.
        random_state: Controls the shuffling applied to the data before applying the split.
        shuffle: Whether or not to shuffle the data before splitting.

    Returns:
        Tuple of train and test dataset.
    """
    assert data.partitions, "Data partitions are None or empty."
    assert 0 < ratio < 1, f"Invalid split ratio {ratio}, must be in (0, 1)"

    if random_state is None:
        random_state = random.randint(0, 2**32 - 1)

    assert isinstance(random_state, int), f"random_state must be an integer"

    def split(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if len(args[0].shape) == 0:
            return np.array(None), np.array(None)
        results = _train_test_split(*args, **kwargs)
        return results[0], results[1]

    parts_train, parts_test = {}, {}
    for device, part in data.partitions.items():
        parts_train[device], parts_test[device] = device(split)(
            part, train_size=ratio, random_state=random_state, shuffle=shuffle
        )
    return (
        FedNdarray(parts_train, data.partition_way),
        FedNdarray(parts_test, data.partition_way),
    )


def shuffle(data: FedNdarray):
    """Random shuffle data.

    Args:
        data: data to be shuffled.
    """
    rng = np.random.default_rng()

    if data.partitions is not None:

        def _shuffle(rng: np.random.Generator, part: np.ndarray):
            new_part = deepcopy(part)
            rng.shuffle(new_part)
            return new_part

        for device, part in data.partitions.items():
            device(_shuffle)(rng, part)


def check_same_partition_shapes(a1: FedNdarray, a2: FedNdarray):
    return (a1.partition_shape() == a2.partition_shape()) and (
        a1.partition_way == a2.partition_way
    )


def unary_op(
    handle_function: Callable,
    len_1_handle_function: Callable,
    y: FedNdarray,
    spu_device: Optional[SPU] = None,
    simulate_double_value_replacer_handle: Optional[Callable] = None,
):
    if simulate_double_value_replacer_handle:
        d = simulate_double_value_replacer_handle(y, spu_device)
    y_len = len(y.partitions.keys())
    if y_len == 1:
        for device, partition in y.partitions.items():
            if simulate_double_value_replacer_handle:
                return device(len_1_handle_function)(partition, d)
            return device(len_1_handle_function)(partition)
    elif y_len > 1:
        assert spu_device is not None, "A SPU device is required"
        obj_list = [to(spu_device, partition) for partition in y.partitions.values()]
        if simulate_double_value_replacer_handle:
            return spu_device(handle_function)(obj_list, d)
        return spu_device(handle_function)(obj_list)
    else:
        return 0


def mean(y: FedNdarray, spu_device: Optional[SPU] = None):
    """Mean of all elements
    Args:
        y: FedNdarray
        spu_device: SPU
    If y is from a single party, then a PYUObject is returned.
    If y is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y is empty return 0.
    """

    def spu_mean(obj_list: List[np.ndarray]):
        sums = jnp.array([jnp.sum(obj) for obj in obj_list])
        sizes = jnp.array([obj.size for obj in obj_list])
        return jnp.sum(sums) / jnp.sum(sizes)

    return unary_op(spu_mean, jnp.mean, y, spu_device)


def binary_op(
    handle_function: Callable,
    len_1_handle_function: Callable,
    y1: FedNdarray,
    y2: FedNdarray,
    spu_device: Optional[SPU] = None,
):
    y_len = len(y1.partitions.keys())
    enable_local_compute_optimization = check_same_partition_shapes(y1, y2)

    if enable_local_compute_optimization:
        if y_len == 1:
            for device, partition in y1.partitions.items():
                return device(len_1_handle_function)(partition, y2.partitions[device])
        elif y_len > 1:
            assert spu_device is not None, "A SPU device is required"
            obj1_list = [
                to(spu_device, partition) for partition in y1.partitions.values()
            ]
            obj2_list = [
                to(spu_device, partition) for partition in y2.partitions.values()
            ]
            return spu_device(handle_function)(obj1_list, obj2_list)
        else:
            return 0
    else:
        assert spu_device is not None, "A SPU device is required"
        assert y1.shape == y2.shape, "Two shapes must coincide"
        obj1_list = [to(spu_device, partition) for partition in y1.partitions.values()]
        obj2_list = [to(spu_device, partition) for partition in y2.partitions.values()]
        axis_1 = get_concat_axis(y1)
        axis_2 = get_concat_axis(y2)

        def binary_op_concat_composition(
            obj1_list: List[np.ndarray],
            obj2_list: List[np.ndarray],
            axis_1: int,
            axis_2: int,
        ):
            n1 = jnp.concatenate(obj1_list, axis=axis_1)
            n2 = jnp.concatenate(obj2_list, axis=axis_2)
            return len_1_handle_function(n1, n2)

        return spu_device(
            binary_op_concat_composition, static_argnames=("axis_1", "axis_2")
        )(obj1_list, obj2_list, axis_1=axis_1, axis_2=axis_2)


def get_concat_axis(y: FedNdarray) -> int:
    if y.partition_way == PartitionWay.HORIZONTAL:
        return 0
    else:
        return 1


def rss(y1: FedNdarray, y2: FedNdarray, spu_device: Optional[SPU] = None):
    """Residual Sum of Squares of all elements

    more detail for rss:
    https://en.wikipedia.org/wiki/Residual_sum_of_squares

    Args:
        y1 : FedNdarray
        y2 : FedNdarray
        spu_device: SPU

    y1 and y2 must have the same device and partition shapes

    If y1 is from a single party, then a PYUObject is returned.
    If y1 is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y1 is empty return 0.

    """

    def spu_rss(obj1_list: List[np.ndarray], obj2_list: List[np.ndarray]):
        sums = jnp.array(
            [
                sum_of_difference_squares(obj1_list[i], obj2_list[i])
                for i in range(len(obj1_list))
            ]
        )
        return jnp.sum(sums)

    return binary_op(spu_rss, sum_of_difference_squares, y1, y2, spu_device)


def tss(y: FedNdarray, spu_device: Optional[SPU] = None):
    """Total Sum of Square (Variance) of all elements

    more detail for tss:
    https://en.wikipedia.org/wiki/Total_sum_of_squares

    Args:
        y: FedNdarray

    If y is from a single party, then a PYUObject is returned.
    If y is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y is empty return 0.
    """

    def spu_tss(obj1_list: List[np.ndarray], y_mean):
        sums = jnp.array([sum_of_difference_squares(obj, y_mean) for obj in obj1_list])
        return jnp.sum(sums)

    return unary_op(
        spu_tss,
        sum_of_difference_squares,
        y,
        spu_device,
        mean,
    )


def mean_squared_error(
    y_true: FedNdarray, y_pred: FedNdarray, spu_device: Optional[SPU] = None
):
    """Mean Squared Error of all elements

    more detail for mse:
    https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        y_true : FedNdarray
        y_pred : FedNdarray
        spu_device: SPU

    y_true and y_pred must have the same device and partition shapes

    If y_true is from a single party, then a PYUObject is returned.
    If y_true is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y_true is empty return 0.

    """

    # TODO: check y1 and y2 have the same device and shapes
    def spu_mse(obj1_list: List[np.ndarray], obj2_list: List[np.ndarray]):
        sums = jnp.array(
            [
                sum_of_difference_squares(obj1_list[i], obj2_list[i])
                for i in range(len(obj1_list))
            ]
        )
        sizes = jnp.array([obj.size for obj in obj2_list])
        return jnp.divide(jnp.sum(sums), jnp.sum(sizes))

    return binary_op(spu_mse, mean_of_difference_squares, y_true, y_pred, spu_device)


def root_mean_squared_error(
    y_true: FedNdarray, y_pred: FedNdarray, spu_device: Optional[SPU] = None
):
    """Root Mean Squared Error of all elements

    more detail for mse:
    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Args:
        y_true : FedNdarray
        y_pred : FedNdarray
        spu_device: SPU

    y_true and y_pred must have the same device and partition shapes

    If y_true is from a single party, then a PYUObject is returned.
    If y_true is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y_true is empty return 0.

    """
    # TODO: check y1 and y2 have the same device and shapes
    def spu_rmse(obj1_list: List[np.ndarray], obj2_list: List[np.ndarray]):
        sums = jnp.array(
            [
                sum_of_difference_squares(obj1_list[i], obj2_list[i])
                for i in range(len(obj1_list))
            ]
        )
        sizes = jnp.array([obj.size for obj in obj2_list])
        return jnp.divide(jnp.sqrt(jnp.sum(sums), jnp.sum(sizes)))

    return binary_op(
        spu_rmse,
        lambda x, y: jnp.sqrt(mean_of_difference_squares(x, y)),
        y_true,
        y_pred,
        spu_device,
    )


def mean_abs_err(
    y_true: FedNdarray, y_pred: FedNdarray, spu_device: Optional[SPU] = None
):
    """Mean Absolute Error

    more detail for mean abs err:
    https://en.wikipedia.org/wiki/Mean_absolute_error

    Args:
        y_true: FedNdarray
        y_pred: FedNdarray

    y_true and y_pred must have the same device and partition shapes

    If y_true is from a single party, then a PYUObject is returned.
    If y_true is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y_true is empty return 0.
    """

    def spu_mean_abs_err(obj1_list: List[np.ndarray], obj2_list: List[np.ndarray]):
        sums = jnp.array(
            [
                sum_of_difference_abs(obj1_list[i], obj2_list[i])
                for i in range(len(obj1_list))
            ]
        )
        sizes = jnp.array([obj.size for obj in obj1_list])
        s = jnp.sum(sizes)
        return jnp.divide(jnp.sum(sums), s)

    return binary_op(
        spu_mean_abs_err, mean_of_difference_abs, y_true, y_pred, spu_device
    )


def mean_abs_percent_err(
    y_true: FedNdarray, y_pred: FedNdarray, spu_device: Optional[SPU] = None
):
    """Mean Absolute Percentage Error

    more detail for mean percent err:
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Args:
        y_true: FedNdarray
        y_pred: FedNdarray

    y_true and y_pred must have the same device and partition shapes

    If y_true is from a single party, then a PYUObject is returned.
    If y_true is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y_true is empty return 0.

    """

    def spu_mean_ratio_abs_err(
        obj1_list: List[np.ndarray], obj2_list: List[np.ndarray]
    ):
        sums = jnp.array(
            [
                sum_of_difference_ratio_abs(obj1_list[i], obj2_list[i])
                for i in range(len(obj1_list))
            ]
        )
        sizes = jnp.array([obj.size for obj in obj1_list])
        s = jnp.sum(sizes)
        return jnp.divide(jnp.sum(sums), s)

    return binary_op(
        spu_mean_ratio_abs_err, mean_of_difference_ratio_abs, y_true, y_pred, spu_device
    )


def r2_score(y_true: FedNdarray, y_pred: FedNdarray, spu_device: Optional[SPU] = None):
    """R2 Score

    more detail for r2 score:
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Args:
        y_true : FedNdarray
        y_pred : FedNdarray

    y_true and y_pred must have the same device and partition shapes

    If y_true is from a single party, then a PYUObject is returned.
    If y_true is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    If y_true is empty return 0.
    """

    def r2_from_tss_rss(tss_val, rss_val):
        return 1 - rss_val / tss_val

    same_p = check_same_partition_shapes(y_true, y_pred)
    y_len = len(y_true.partitions.keys())
    tss_val = tss(y_true, spu_device)
    rss_val = rss(y_true, y_pred, spu_device)

    if same_p and y_len == 1:
        for device in y_true.partitions.keys():
            return device(r2_from_tss_rss)(tss_val, rss_val)
    else:
        assert spu_device is not None, "A SPU device is required"
        return spu_device(r2_from_tss_rss)(
            to(spu_device, tss_val), to(spu_device, rss_val)
        )


def histogram(y: FedNdarray, bins: int = 10, spu_device: Optional[SPU] = None):
    """Histogram of all elements
    a restricted version of the counterpart in numpy

    more detail for histogram:
    https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

    Args:
        y: FedNdarray

    If y is from a single party, then a PYUObject is returned.
    If y is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    """
    y_len = len(y.partitions.keys())
    if y_len == 1:
        for device, partition in y.partitions.items():
            return device(jnp.histogram)(partition, bins)
    else:
        assert False, "Feature Not Supported Yet"
        assert spu_device is not None, "A SPU device is required"
        obj_list = [to(spu_device, partition) for partition in y.partitions.values()]
        axis = get_concat_axis(y)

        def hist_concat_composition(obj_list: List[np.array], axis: int, bins: int):
            return jnp.histogram(jnp.concatenate(obj_list, axis=axis), bins=bins)

        return spu_device(hist_concat_composition, static_argnames=("bins", "axis"))(
            obj_list, bins=bins, axis=axis
        )


def residual_histogram(
    y1: FedNdarray, y2: FedNdarray, bins: int = 10, spu_device: Optional[SPU] = None
):
    """Histogram of residuals of  y1 - y2

    Support histogram(y1 - y2) equivalent function even if y1 and y2 have distinct partition shapes.

    Args:
        y1: FedNdarray
        y2: FedNdarray

    If y is from a single party, then a PYUObject is returned.
    If y is from multiple parties, then
        a SPU device is required and a SPUObject is returned.
    """
    y_len = len(y1.partitions.keys())
    enable_local_compute_optimization = check_same_partition_shapes(y1, y2)

    def hist_subtract_composition(y1, y2, bins, axis=None):
        if type(y1) == type(list):
            residual = jnp.concatenate([y1[i] - y2[i] for i in range(len(y1))], axis)
        else:
            residual = jnp.subtract(y1, y2)
        return jnp.histogram(residual, bins=bins)

    if enable_local_compute_optimization:
        if y_len == 1:
            for device, partition in y1.partitions.items():
                return device(hist_subtract_composition, static_argnames="bins")(
                    partition, y2.partitions[device], bins=bins
                )
        elif y_len > 1:
            assert False, "Feature Not Supported Yet"
            assert spu_device is not None, "A SPU device is required"
            obj1_list = [
                to(spu_device, partition) for partition in y1.partitions.values()
            ]
            obj2_list = [
                to(spu_device, partition) for partition in y2.partitions.values()
            ]
            return spu_device(
                hist_subtract_composition, static_argnames=("bins", "axis")
            )(obj1_list, obj2_list, bins=bins, axis=get_concat_axis(y1))
        else:
            # empty FedNdarray
            return None
    else:
        assert False, "Feature Not Supported Yet"
        assert spu_device is not None, "A SPU device is required"
        assert y1.shape == y2.shape, "Two shapes must coincide"
        obj1_list = [to(spu_device, partition) for partition in y1.partitions.values()]
        obj2_list = [to(spu_device, partition) for partition in y2.partitions.values()]
        axis_1 = get_concat_axis(y1)
        axis_2 = get_concat_axis(y2)

        def hist_concat_composition(
            obj1_list: List[np.ndarray],
            obj2_list: List[np.ndarray],
            axis_1: int,
            axis_2: int,
            bins: int,
        ):
            n1 = jnp.concatenate(obj1_list, axis=axis_1)
            n2 = jnp.concatenate(obj2_list, axis=axis_2)
            return jnp.histogram(n1 - n2, bins=bins)

        return spu_device(
            hist_concat_composition, static_argnames=("axis_1", "axis_2", "bins")
        )(obj1_list, obj2_list, axis_1=axis_1, axis_2=axis_2, bins=bins)
