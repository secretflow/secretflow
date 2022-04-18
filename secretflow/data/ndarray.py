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
from typing import Callable, Dict, List, Tuple, Union
import functools
import operator

import numpy as np
from sklearn.model_selection import train_test_split as _train_test_split

from secretflow.data.io import util as io_util
from secretflow.device.device import PYU, PYUObject
from secretflow.device.device.base import reveal
from secretflow.utils.errors import InvalidArgumentError


# TODO @饶飞： 暂时没有区分是水平还是垂直。下面的函数是同时支持水平和垂直的。
__ndarray = "__ndarray_type__"


@dataclass
class FedNdarray:
    partitions: Dict[PYU, PYUObject]

    @reveal
    def length(self):
        return {device: device(lambda partition: len(partition))(partition) for device, partition in
                self.partitions.items()}


def load(sources: Dict[PYU, Union[str, Callable[[], np.ndarray], PYUObject]],
         allow_pickle=False, encoding='ASCII') -> FedNdarray:
    """加载指定的数据构造FedNdarray。
    Args:
        sources: 字典形式的数据来源，key是参与方id，value表示ndarray的来源，可以是字符串或者函数或者ndarray。
          若value为字符串，则可以是本地文件路径，网络文件url或者pathlib.Path。
          若value为函数，则函数应该返回ndarray。
        allow_pickle: 可选; 同numpy.load函数的allow_pickle参数。
        encoding: 可选; 同numpy.load函数的encoding参数。

    Raises:
        TypeError: source类型非法。

    Returns:
        case1: pyuobject -> FedNdarray对象
        case2: .npy -> FedNdarray对象
        case3: .npz -> Dict{key:FedNdarray}

    Examples
    -------
    >>> fed_arr = load({'alice': 'example/alice.csv', 'bob': 'example/alice.csv'})
    """

    def _load(content) -> (List, List):
        if isinstance(content, str):
            data = np.load(io_util.open(content),
                           allow_pickle=allow_pickle, encoding=encoding)
        elif isinstance(content, Callable):
            data = content()
        else:
            raise TypeError(f"Unsupported source with {type(content)}.")
        assert isinstance(data, np.ndarray) or isinstance(
            data, np.lib.npyio.NpzFile)
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
            raise InvalidArgumentError(
                'Device of source differs with its key.')
        if not isinstance(content, PYUObject):
            files, datas = device(_load)(content)
            file_list.append(reveal(files))
            data_dict[device] = datas
        else:
            pyu_parts[device] = content
    # 处理pyu object
    if pyu_parts:
        return FedNdarray(partitions=pyu_parts)

    # 检查各方的数据是否一致
    is_same = [file_list[i-1] == file_list[i]
               for i in range(1, len(file_list))]
    check_status = functools.reduce(operator.and_, is_same)
    if not check_status:
        raise Exception(
            f"All parties should have same structure,but got file_list = {file_list}")
    file_names = file_list[0]
    result = {}
    for idx, m in enumerate(file_names):
        parts = {}
        for device, data in data_dict.items():
            parts[device] = device(_get_item)(idx, data)
        if m == __ndarray and len(file_names) == 1:
            return FedNdarray(partitions=parts)
        result[m] = FedNdarray(partitions=parts)
    return result


def train_test_split(data: FedNdarray, ratio: float,
                     random_state: int = None, shuffle=True) -> Tuple[FedNdarray, FedNdarray]:
    """将FedNdarray拆分成训练和测试数据集。

    Args:
        data: 待拆分的FedNdarray。
        ratio: 拆分比例，必须是(0, 1)之间的某个值。
        random_state: int, 随机种子。如果为None，则随机生成。
        shuffle: 拆分前是否打乱顺序，默认为true。

    Returns:
        由训练和测试FedNdarray组成的元组。
    """
    assert data.partitions, 'Data partitions are None or empty.'
    assert 0 < ratio < 1, f"Invalid split ratio {ratio}, must be in (0, 1)"

    if random_state is None:
        random_state = random.randint(0, 2 ** 32 - 1)

    assert isinstance(random_state, int), f'random_state must be an integer'

    def split(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if len(args[0].shape) == 0:
            return np.array(None), np.array(None)
        results = _train_test_split(*args, **kwargs)
        return results[0], results[1]

    parts_train, parts_test = {}, {}
    for device, part in data.partitions.items():
        parts_train[device], parts_test[device] = device(split)(
            part, train_size=ratio, random_state=random_state, shuffle=shuffle)
    return FedNdarray(parts_train), FedNdarray(parts_test)


def shuffle(data: FedNdarray):
    """对FedNdarray进行顺序打乱。

    Args:
        data: 待处理的FedNdarray。
    """
    rng = np.random.default_rng()

    if data.partitions is not None:
        def _shuffle(rng: np.random.Generator, part: np.ndarray):
            new_part = deepcopy(part)
            rng.shuffle(new_part)
            return new_part

        for device, part in data.partitions.items():
            device(_shuffle)(rng, part)
