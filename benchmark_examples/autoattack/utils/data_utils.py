# Copyright 2023 Ant Group Co., Ltd.
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

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from secretflow import PYU, PYUObject, reveal
from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame


class SparseTensorDataset(Dataset):
    def __init__(self, x: List, indexes: np.ndarray = None, **kwargs):
        data = x[0]
        if indexes is not None:
            data = data[indexes]
        self.tensors = []
        for col in range(data.shape[1]):
            self.tensors.append(torch.tensor(data[:, col], dtype=torch.int64))
        self.label = None
        if len(x) > 1:
            label = x[1]
            if indexes is not None:
                label = label[indexes]
            self.label = torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, index):
        if self.label is not None:
            return tuple(tensor[index] for tensor in self.tensors), self.label[index]
        else:
            return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class CustomTensorDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        label: np.ndarray | None = None,
        enable_label: int = 0,
        **kwargs,
    ):
        # 0 true 1 false -1:-1
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = None
        self.enable_label = enable_label
        if label is not None and enable_label == 0:
            self.label = torch.tensor(label, dtype=torch.int64)

    def __getitem__(self, item):
        if self.enable_label == 0 and self.label is not None:
            return self.data[item], self.label[item]
        elif self.enable_label == -1:
            return self.data[item], torch.tensor(-1)
        else:
            return (self.data[item],)

    def __len__(self):
        return self.data.size(0)


def create_custom_dataset_builder(
    CustomDataSetClass: type(Dataset), batch_size, **kwargs
):
    """
    If giving a standard dataset, with which the __init__ params contains x just like the dataset builder need,
    then we can use this function to generate a custom dataset builder.
    Args:
        CustomDataSetClass: A class inherits from Dataset.
        batch_size: batch size.
        **kwargs:

    Returns:
        A dataset builder.
    """

    def dataset_builder(x):
        import torch.utils.data as torch_data

        data_set = CustomDataSetClass(x, **kwargs)
        dataloader = torch_data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def get_sample_indexes(
    length: int, sample_size: int = None, frac: float = None, indexes: np.ndarray = None
) -> np.ndarray | None:
    """
    Get sample indexes.
    Args:
        length: total length.
        sample_size: sample size.
        frac: frac.
        indexes: sample indexes.
    Returns:
        the sample indexes, None for all.
    """
    assert (
        sum(x is not None for x in [sample_size, frac, indexes]) <= 1
    ), f"sample_ndarray need only one argument (sample_size {sample_size}, frac {frac} or indexes {indexes}), got muti or all None."
    if sample_size is not None:
        assert sample_size <= length, f"Sample size {sample_size} > length {length}"
        if sample_size == length:
            return None
        return np.random.choice(length, size=sample_size, replace=False)
    if frac is not None:
        assert 0 < frac <= 1, f"frac {frac} must be between 0 and 1"
        if frac == 1:
            return None
        sample_size = int(length * frac)
        return get_sample_indexes(length, sample_size=sample_size)
    if indexes is not None:
        assert len(indexes) < length, f"len of indexes {len(indexes)} > length {length}"
        return indexes
    return None


def sample_ndarray(
    arrays,
    sample_size: int = None,
    frac: float = None,
    indexes: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Sample data from arrays according to sample_size or frac.
    Args:
        arrays: input arrays
        sample_size: sample number.
        frac: frac
        indexes: sample indexes.

    Returns:
        A tuple of sampled data and indexes.
    """
    indexes = get_sample_indexes(arrays.shape[0], sample_size, frac, indexes)
    if indexes is None:
        return arrays, None
    return arrays[indexes], indexes


def reveal_data(data: VDataFrame | FedNdarray | PYUObject) -> np.ndarray:
    if isinstance(data, VDataFrame):
        return reveal_data(data.values)
    elif isinstance(data, FedNdarray):
        if len(data.partitions) > 1:
            return np.concatenate([reveal(d) for d in data.partitions.values()], axis=1)
        else:
            x = reveal(list(data.partitions.values())[0])
            return x
    elif isinstance(data, PYUObject):
        return reveal(data)
    else:
        raise TypeError(f"Unsuport reveal data with type = {type(data)}")


def reveal_part_data(
    data: VDataFrame | FedNdarray | PYUObject, part: PYU
) -> np.ndarray:
    if isinstance(data, VDataFrame):
        return reveal(data.partitions[part].data).values
    elif isinstance(data, FedNdarray):
        return reveal(data.partitions[part])
    elif isinstance(data, PYUObject):
        assert (
            data.device == part
        ), f"data.device = {data.device} while try to get part = {part}"
        return reveal(data)
    else:
        raise TypeError(f"Unsuport reveal data with type = {type(data)}")


def get_np_data_from_dataset(dataset: Dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))
    data = [data] if not isinstance(data, (List, Tuple)) else data
    data = [d.numpy() for d in data]
    if len(data) == 1:
        return data[0]
    return data
