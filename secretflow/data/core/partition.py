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
from typing import Callable, List, Union

from secretflow.device import PYU, Device, PYUObject, reveal
from secretflow.distributed.primitive import get_current_cluster_idx

from ..base import DataFrameBase
from .agent import PartitionAgent
from .base import AgentIndex, PartitionAgentBase


def partition(
    data: Union[Callable, PYUObject],
    device: Device = None,
    backend="pandas",
    **kwargs,
) -> "Partition":
    """
    Construct a Partition with input data.
    Notes: This function should not be called multiple times.
    Args:
        data: The source to construct a partiton.
            It can be a Callable func type which takes kwargs as parameters and retures a real dataframe.
            or, it can be a PYUObject within a real dataframe type.
            We suggest to use function source to improve performance.
        device: Which device use this partition.
        backend: The partition backend, default to pandas.
        kwargs: if the source is a function, use kwargs as its parameters.
    Returns:
        A Partition.
    """
    if callable(data):
        assert device is not None, f"cannot infer device from a callable source."
    else:
        assert (
            device is None or device == data.device
        ), f"source's device does not match input device."
        device = data.device
        logging.warning("To create a Partitoin, we suggest to use function source.")
    agent: PartitionAgentBase = PartitionAgent(device=device)
    index: AgentIndex = agent.append_data(data, backend, **kwargs)
    return Partition(part_agent=agent, agent_idx=index, device=device, backend=backend)


class StatPartition:
    """
    A wrapper of pd.Series.
    Generally, it will be used to wrap a statistic results, which is pd.Series type.
    """

    data: PYUObject

    def __init__(self, data: PYUObject):
        self.data = data

    @property
    def index(self) -> PYUObject:
        # inside it returns list
        return reveal(self.data.device(lambda series: series.index)(self.data))

    @property
    def values(self) -> PYUObject:
        # inside it returns np.ndarray
        return self.data.device(lambda series: series.values)(self.data)


class Partition(DataFrameBase):
    """Partition of a party"""

    part_agent: PartitionAgentBase
    agent_idx: Union[AgentIndex, PYUObject]
    device: PYU
    backend: str
    active_cluster_idx: int

    def __init__(
        self,
        part_agent: PartitionAgentBase,
        agent_idx: Union[AgentIndex, PYUObject],
        device: PYU,
        backend="pandas",
    ):
        """
        A party's partition, which use a part_agent wrapper to access real dataframe.
        Args:
            part_agent: A PYU actor which resident on one party. Generally, to ensure processing performance,
                only create a part_agent when initializing data (such as read_csv).
            agent_idx: A index to access real data inside the part_agent.
            device: The PYU device.
            backend: The read backend, default to pandas.
        """
        self.part_agent: PartitionAgentBase = part_agent
        self.agent_idx = agent_idx
        self.device = device
        self.backend: str = backend
        self.active_cluster_idx: int = get_current_cluster_idx()

    def __del__(self):
        """
        When a partition id deleted, we should make shure that the data corresponding to partition's agent_idx
        in the part_agent actor is also deleted. So we need to explicit call the del_object ont eh part_agent.
        However, in following cases, when the varible 'temp' is deleted, the first cluster is already shutdown,
        calling the 'del_object()' on the actor in a died cluster will cause unrecoverable errors.
        Therefore, the partition record the cluter index to which it belongs.
        When the current active cluster is not which the partition belongs, the __del__ do nothing.
            - sf.init()
            - temp = Partition(xx)
            - sf.shutdown()
            - sf.init()
            - temp = Partition(xx) # temp will be deleted first.
            - sf.shutdown()
        """
        if get_current_cluster_idx() == self.active_cluster_idx:
            self.part_agent.del_object(self.agent_idx)

    def __getitem__(self, item) -> "Partition":
        data_idx = self.part_agent.__getitem__(self.agent_idx, item)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def __setitem__(self, key, value: Union['Partition', PYUObject]):
        if isinstance(value, (PYUObject, Partition)):
            assert (
                self.device == value.device
            ), f'Can not assign a partition with different device.'
        if isinstance(value, Partition):
            if self.part_agent == value.part_agent:
                # same part_agent directly set with index.
                self.part_agent.__setitem__(self.agent_idx, key, value.agent_idx)
            else:
                # different part_agent need to get data and then set into it.
                self.part_agent.__setitem__(
                    self.agent_idx, key, value.part_agent.get_data(value.agent_idx)
                )
        else:
            self.part_agent.__setitem__(self.agent_idx, key, value)

    def __len__(self):
        return reveal(self.part_agent.__len__(self.agent_idx))

    @property
    def columns(self) -> list:
        return reveal(self.part_agent.columns(self.agent_idx))

    @property
    def dtypes(self) -> dict:
        return reveal(self.part_agent.dtypes(self.agent_idx))

    @property
    def shape(self) -> tuple:
        return reveal(self.part_agent.shape(self.agent_idx))

    @property
    def index(self) -> list:
        return reveal(self.part_agent.index(self.agent_idx))

    def count(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.count(self.agent_idx, *args, **kwargs))

    def sum(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.sum(self.agent_idx, *args, **kwargs))

    def min(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.min(self.agent_idx, *args, **kwargs))

    def max(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.max(self.agent_idx, *args, **kwargs))

    def mean(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.mean(self.agent_idx, *args, **kwargs))

    def var(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.var(self.agent_idx, *args, **kwargs))

    def std(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.std(self.agent_idx, *args, **kwargs))

    def sem(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.sem(self.agent_idx, *args, **kwargs))

    def skew(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.skew(self.agent_idx, *args, **kwargs))

    def kurtosis(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.kurtosis(self.agent_idx, *args, **kwargs))

    def quantile(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.quantile(self.agent_idx, *args, **kwargs))

    def mode(self, *args, **kwargs) -> StatPartition:
        return StatPartition(self.part_agent.mode(self.agent_idx, *args, **kwargs))

    def value_counts(self, *args, **kwargs) -> StatPartition:
        return StatPartition(
            self.part_agent.value_counts(self.agent_idx, *args, **kwargs)
        )

    @property
    def values(self) -> PYUObject:
        # Will return a PYUObject within np.ndarray type.
        return self.part_agent.values(
            self.agent_idx,
        )

    def isna(self) -> "Partition":
        data_idx = self.part_agent.isna(self.agent_idx)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def replace(self, *args, **kwargs) -> "Partition":
        data_idx = self.part_agent.replace(self.agent_idx, *args, **kwargs)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def astype(self, dtype, copy: bool = True, errors: str = "raise") -> "Partition":
        data_idx = self.part_agent.astype(self.agent_idx, dtype, copy, errors)
        if copy:
            return Partition(self.part_agent, data_idx, self.device, self.backend)

    def copy(self) -> "Partition":
        data_idx = self.part_agent.copy(self.agent_idx)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ) -> "Partition":
        data_idx = self.part_agent.drop(
            self.agent_idx, labels, axis, index, columns, level, inplace, errors
        )
        if not inplace:
            return Partition(self.part_agent, data_idx, self.device, self.backend)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['Partition', None]:
        data_idx = self.part_agent.fillna(
            self.agent_idx, value, method, axis, inplace, limit, downcast
        )
        if not inplace:
            return Partition(self.part_agent, data_idx, self.device, self.backend)

    def to_csv(self, filepath, **kwargs):
        self.part_agent.to_csv(self.agent_idx, filepath, **kwargs)

    def iloc(self, index: Union[int, slice, List[int]]) -> 'Partition':
        data_idx = self.part_agent.iloc(self.agent_idx, index)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors='ignore',
    ) -> Union['Partition', None]:
        data_idx = self.part_agent.rename(
            self.agent_idx, mapper, index, columns, axis, copy, inplace, level, errors
        )
        if not inplace:
            return Partition(self.part_agent, data_idx, self.device, self.backend)

    def pow(self, *args, **kwargs) -> 'Partition':
        data_idx = self.part_agent.pow(self.agent_idx, *args, **kwargs)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def round(self, *args, **kwargs) -> 'Partition':
        data_idx = self.part_agent.round(self.agent_idx, *args, **kwargs)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def select_dtypes(self, *args, **kwargs) -> 'Partition':
        data_idx = self.part_agent.select_dtypes(self.agent_idx, *args, **kwargs)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def subtract(self, other) -> 'Partition':
        sub_other = other
        if isinstance(other, (StatPartition)):
            sub_other = other.data
        elif isinstance(other, Partition):
            assert self.device == other.device
            sub_other = other.agent_idx
        data_idx = self.part_agent.subtract(self.agent_idx, sub_other)
        return Partition(self.part_agent, data_idx, self.device, self.backend)

    def apply_func(
        self, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> Union['Partition', 'tuple[Partition]']:
        data_idx = self.part_agent.apply_func(
            self.agent_idx, func, nums_return=nums_return, **kwargs
        )
        if nums_return == 1:
            data_idx = reveal(data_idx)
            return Partition(
                self.part_agent,
                data_idx,
                self.device,
                reveal(self.part_agent.get_backend(data_idx)),
            )
        else:
            data_idx_list = reveal(data_idx)
            assert (
                isinstance(data_idx_list, list) and len(data_idx_list) == nums_return
            ), (
                f"nums_return not match, got type = {type(data_idx_list)}, len = {len(data_idx_list)} "
                f"while nums_return = {nums_return}"
            )
            return tuple(
                [
                    Partition(
                        self.part_agent,
                        idx,
                        self.device,
                        reveal(self.part_agent.get_backend(idx)),
                    )
                    for idx in data_idx_list
                ]
            )

    def to_pandas(self) -> 'Partition':
        if self.backend == "pandas":
            return self
        else:
            data_idx = self.part_agent.to_pandas(self.agent_idx)
            return Partition(self.part_agent, data_idx, self.device, self.backend)

    @property
    def data(self):
        return self.part_agent.get_data(self.agent_idx)
