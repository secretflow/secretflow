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

from typing import Tuple, Union

import numpy as np
import pandas as pd

import logging
from sklearn.model_selection import train_test_split as _train_test_split

from secretflow.data.base import Partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical.dataframe import VDataFrame


def train_test_split(
    data: Union[VDataFrame, HDataFrame, FedNdarray],
    test_size=None,
    train_size=None,
    random_state=1234,
    shuffle=True,
    stratify=None,
) -> Tuple[object, object]:
    """Split data into train and test dataset.

    Args:
        data : DataFrame to split, supported are: VDataFrame,HDataFrame,FedNdarray.
        test_size (float): test dataset size, default is None.
        train_size (float): train dataset size, default is None.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        shuffle (bool): Whether or not to shuffle the data before splitting, default is True.
        stratify (array-like): If not None, data is split in a stratified fashion, using this as the class labels.

    Returns
        splitting : list, length=2 * len(arrays)

    Examples:
        >>> import numpy as np
        >>> from secret.data.split import train_test_split
        >>> # FedNdarray
        >>> alice_arr = alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
        >>> bob_arr = bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

        >>> fed_arr = load({self.alice: alice_arr, self.bob: bob_arr})
        >>>
        >>> X_train, X_test = train_test_split(
        ...  fed_arr, test_size=0.33, random_state=42)
        ...
        >>> VDataFrame
        >>> df_alice = pd.DataFrame({'a1': ['K5', 'K1', None, 'K6'],
        ...                          'a2': ['A5', 'A1', 'A2', 'A6'],
        ...                          'a3': [5, 1, 2, 6]})

        >>> df_bob = pd.DataFrame({'b4': [10.2, 20.5, None, -0.4],
        ...                        'b5': ['B3', None, 'B9', 'B4'],
        ...                        'b6': [3, 1, 9, 4]})
        >>> df_alice = df_alice
        >>> df_bob = df_bob
        >>> vdf = VDataFrame(
        ...       {alice: Partition(data=cls.alice(lambda: df_alice)()),
        ...          bob: Partition(data=cls.bob(lambda: df_bob)())})
        >>> train_vdf, test_vdf = train_test_split(vdf, test_size=0.33, random_state=42)

    """
    assert type(data) in [HDataFrame, VDataFrame, FedNdarray]
    assert data.partitions, 'Data partitions are None or empty.'
    if test_size is not None and train_size is None:
        assert 0 < test_size < 1, f"Invalid test size {test_size}, must be in (0, 1)"
    elif test_size is None and train_size is not None:
        assert 0 < train_size < 1, f"Invalid train size {train_size}, must be in (0, 1)"
    elif test_size is not None and train_size is not None:
        test_size = None
        logging.info(
            "Neither train_size nor test_size is empty, Here use train_size for split"
        )
    else:
        raise Exception("invalid params")

    assert isinstance(random_state, int), f'random_state must be an integer'

    def split(*args, **kwargs) -> Tuple[object, object]:
        assert type(args[0]) in [np.ndarray, pd.DataFrame]
        if len(args[0].shape) == 0:
            if type(args[0]) == np.ndarray:
                return np.array(None), np.array(None)
            else:
                return pd.DataFrame(None), pd.DataFrame(None)
        results = _train_test_split(*args, **kwargs)
        return results[0], results[1]

    parts_train, parts_test = {}, {}
    for device, part in data.partitions.items():
        if isinstance(data, FedNdarray):
            part_data = part
        else:
            part_data = part.data
        parts_train[device], parts_test[device] = device(split)(
            part_data,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

    if isinstance(data, VDataFrame):
        return VDataFrame(
            partitions={pyu: Partition(data=part) for pyu, part in parts_train.items()},
            aligned=data.aligned,
        ), VDataFrame(
            partitions={pyu: Partition(data=part) for pyu, part in parts_test.items()},
            aligned=data.aligned,
        )
    elif isinstance(data, HDataFrame):
        return HDataFrame(
            partitions={pyu: Partition(data=part) for pyu, part in parts_train.items()},
            aggregator=data.aggregator,
            comparator=data.comparator,
        ), HDataFrame(
            partitions={pyu: Partition(data=part) for pyu, part in parts_test.items()},
            aggregator=data.aggregator,
            comparator=data.comparator,
        )
    else:
        return (
            FedNdarray(parts_train, data.partition_way),
            FedNdarray(parts_test, data.partition_way),
        )
