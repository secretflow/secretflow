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

import numpy as np
from torch.utils.data import Dataset


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


def sample_ndarray(arrays, frac=0.4):
    """Sample data from arrays according to frac."""
    if frac == 1.0:
        return arrays
    else:
        sample_indexes = np.random.choice(
            arrays.shape[0], size=int(arrays.shape[0] * frac), replace=False
        )
        return arrays[sample_indexes]
