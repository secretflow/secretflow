# Copyright 2024 Ant Group Co., Ltd.
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
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from secretflow.device import Device, DeviceObject, PYUObject, reveal
from secretflow.utils.blocked_ops import block_compute, cut_device_object

# In all of the following functions,
# we assue arrs are from multiple PYUs,
# and each of them is flattened i.e. shape (-1,)
# If no warning then all computations should satisfy
# the secure multi-party computation definition:
# no information is revealed to any other party except for the final result
# and the information can be deduced from the final result and the array size.


def compute_sizes(arrs: List[PYUObject]) -> List[DeviceObject]:
    """Compute the size of each array."""
    return [obj.device(lambda x: x.size)(obj) for obj in arrs]


# array size protected
def size_weights(
    arrs: List[PYUObject], compute_device: Device, sizes: List[PYUObject] = None
) -> DeviceObject:
    """Compute the weight of each array based on its size."""
    if sizes is None:
        sizes = compute_sizes(arrs)
    weights = compute_device(lambda sizes: jnp.array(sizes) / sum(sizes))(sizes)
    return weights


# array size protected
def united_mean(
    arrs: List[PYUObject], compute_device: Device, weights: DeviceObject = None
) -> DeviceObject:
    """Compute the mean of multiple arrays."""
    if weights is None:
        weights = size_weights(arrs, compute_device)
    means = [obj.device(lambda x: x.mean())(obj) for obj in arrs]
    result = compute_device(
        lambda weights, means: sum([w * m for w, m in zip(weights, means)])
    )(weights, means)
    return result


# array size protected
def united_mean_and_var(
    arrs: List[PYUObject], compute_device: Device
) -> Tuple[DeviceObject, DeviceObject]:
    weights = size_weights(arrs, compute_device)
    mean_val = united_mean(arrs, compute_device, weights)
    residual_squares = [
        obj.device(lambda x, mean_val: (x - mean_val) ** 2)(
            obj, mean_val.to(obj.device)
        )
        for obj in arrs
    ]
    result = united_mean(residual_squares, compute_device, weights)
    return mean_val, result


# array size not protected
def united_var(
    arrs: List[PYUObject], compute_device: Device, block_size: int = 100000
) -> DeviceObject:
    """Compute the variance of multiple arrays without revealing mean"""
    sizes = reveal(compute_sizes(arrs))
    weights = size_weights(arrs, compute_device, sizes)
    mean_val = united_mean(arrs, compute_device, weights)

    def residual_sum_of_square(inputs):
        x, mean_val = inputs
        return jnp.sum((x - mean_val) ** 2)

    residual_sum_of_squares = []
    for obj in arrs:
        residual_sum_of_squares.append(
            block_compute(
                cut_device_object(obj, block_size, compute_device, mean_val),
                compute_device,
                residual_sum_of_square,
                lambda x, y: x + y,
            )
        )

    result = compute_device(
        lambda weights, residual_sum_of_squares, sizes: sum(
            [w * m / s for w, m, s in zip(weights, residual_sum_of_squares, sizes)]
        )
    )(weights, residual_sum_of_squares, sizes)
    return result


# array size protected
def _sort(arrs: List[PYUObject]) -> List[PYUObject]:
    return [obj.device(lambda arr: np.sort(arr))(obj) for obj in arrs]


# array size not protected
# from LeetCode "find the median in two sorted arrays"
# log(min(m, n)) time complexity
# WARNING: this is not MPC safe when array elements contain repetitions
def united_median(
    arrs: List[PYUObject], compute_device: Device, sorted: bool = False
) -> DeviceObject:
    """Compute the median in side a list of arrays.

    Args:
        arrs (List[PYUObject]): List of PYUObject which are in fact arrays in different PYU devices.
        sorted (bool, optional): whether these arrays are already sorted. Defaults to False.

    Returns:
        DeviceObject: median in device.
    """
    # currently assume there are only 2 parties
    assert 2 == len(arrs), 'united_median requires exactly two parties'

    sizes = reveal([obj.device(lambda x: x.size)(obj) for obj in arrs])
    if sizes[0] > sizes[1]:
        arrs[0], arrs[1] = arrs[1], arrs[0]
        sizes[0], sizes[1] = sizes[1], sizes[0]
    if not sorted:
        arrs = _sort(arrs)

    infinty = np.inf
    m, n = sizes[0], sizes[1]
    left, right = 0, m
    # median1：maximum of the left partion
    # median2：minimum of the right partion
    median1, median2 = 0, 0

    while left <= right:
        # left partition includes nums1[0 .. i-1] and nums2[0 .. j-1]
        # right partition includes nums1[i .. m-1] and nums2[j .. n-1]
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i
        # nums_im1, nums_i, nums_jm1, nums_j represents nums1[i-1], nums1[i], nums2[j-1], nums2[j]
        nums_im1 = (
            -infinty
            if i == 0
            else arrs[0]
            .device(lambda arr, i: arr[i])(arrs[0], i - 1)
            .to(compute_device)
        )
        nums_i = (
            infinty
            if i == m
            else arrs[0].device(lambda arr, i: arr[i])(arrs[0], i).to(compute_device)
        )
        nums_jm1 = (
            -infinty
            if j == 0
            else arrs[1]
            .device(lambda arr, i: arr[i])(arrs[1], j - 1)
            .to(compute_device)
        )
        nums_j = (
            infinty
            if j == n
            else arrs[1].device(lambda arr, i: arr[i])(arrs[1], j).to(compute_device)
        )

        if reveal(compute_device(lambda a, b: jnp.less_equal(a, b))(nums_im1, nums_j)):
            median1, median2 = compute_device(lambda a, b: jnp.maximum(a, b))(
                nums_im1, nums_jm1
            ), compute_device(lambda a, b: jnp.minimum(a, b))(nums_i, nums_j)
            left = i + 1
        else:
            right = i - 1

    return (
        compute_device(lambda x, y: (x + y) / 2)(median1, median2)
        if (m + n) % 2 == 0
        else median1
    )
