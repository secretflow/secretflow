# Copyright 2023 Ant Group Co., Ltd.
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
# limitations under the License
from typing import List, Union

import jax.numpy as jnp
import numpy as np

from secretflow import Device, DeviceObject, PYUObject, SPUObject, reveal
from secretflow.data.vertical import VDataFrame


def block_compute(blocked_inputs, compute_device, block_func, aggregation_func):
    agg_result = None
    for block in blocked_inputs:
        # assumed block is already in compute device
        block_result = compute_device(block_func)(block)
        if agg_result is None:
            agg_result = block_result
        else:
            agg_result = compute_device(aggregation_func)(agg_result, block_result)
    return agg_result


def stateful_block_compute(
    blocked_inputs, compute_device, block_func, aggregation_func, initial_state
):
    agg_result = None
    for block in blocked_inputs:
        # assumed block is already in compute device
        block_result = compute_device(block_func)(block)
        if agg_result is None:
            agg_result = compute_device(aggregation_func)(initial_state, block_result)
        else:
            agg_result = compute_device(aggregation_func)(agg_result, block_result)
    return agg_result


def stack_vdata_blocks(blocks: List[np.ndarray]):
    return jnp.concatenate(blocks, axis=1)


def cut_vdata(
    vdf: Union[VDataFrame, List[DeviceObject]],
    row_size: int,
    target_device: Device,
    pad_ones: bool = False,
):
    if isinstance(vdf, VDataFrame):
        vdf = [part.data for part in vdf.partitions.values()]
    assert isinstance(vdf, list), f"{vdf}"
    assert len(vdf) > 0
    assert isinstance(vdf[0], PYUObject), "only support pyu object now"
    m = reveal(vdf[0].device(lambda x: x.shape[0])(vdf[0]))

    for i in range(0, m, row_size):
        end = min(i + row_size, m)
        blocks = [
            device_block.device(lambda x, i, end: x[i:end])(device_block, i, end).to(
                target_device
            )
            for device_block in vdf
        ]
        if pad_ones:
            yield target_device(stack_vdata_blocks)([*blocks, jnp.ones((end - i, 1))])
        else:
            yield target_device(stack_vdata_blocks)(blocks)


def cut_device_object(
    obj, row_size: int, target_device: Device, add_constant: DeviceObject = None
):
    m = reveal(obj.device(lambda x: x.shape[0])(obj))
    for i in range(0, m, row_size):
        end = min(i + row_size, m)
        if isinstance(obj, SPUObject):
            block = obj.device(
                lambda x, i, end: x[i:end], static_argnames=['i', 'end']
            )(obj, i=i, end=end)
        else:
            block = obj.device(lambda x, i, end: x[i:end])(obj, i=i, end=end)
        if add_constant is not None:
            yield target_device(lambda x: x)(block.to(target_device)), add_constant
        else:
            yield target_device(lambda x: x)(block.to(target_device))


def block_compute_vdata(
    vdf: Union[VDataFrame, List[DeviceObject]],
    row_size: int,
    compute_device: Device,
    block_func,
    aggregation_func,
    pad_ones: bool = False,
) -> DeviceObject:
    """Blocked computation for vertical dataframe or equivalent list.
    Blocked computation for vertical dataframe or equivalent list.
    Args:
        vdf (VDataFrame): vertical dataframe or equivalent list
        row_size (int): block size
        compute_device (Device): compute device
        block_func (function): block function
        aggregation_func (function): aggregation function
        pad_ones (bool, optional): whether to pad ones. Defaults to False.
    Returns:
        result (DeviceObject): result
    """
    return block_compute(
        cut_vdata(vdf, row_size, compute_device, pad_ones),
        compute_device,
        block_func,
        aggregation_func,
    )
