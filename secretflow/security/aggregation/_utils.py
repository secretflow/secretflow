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

from typing import List


def is_nesting_list(data: List) -> bool:
    assert data, 'Data should not be None or empty.'
    is_list = isinstance(data[0], list)
    for datum in data[1:]:
        assert (
            isinstance(datum, list) == is_list
        ), f'Some data are list while some others are not.'
        assert not is_list or len(datum) == len(
            data[0]
        ), f'Lengths of datum in data are different.'
    return is_list
