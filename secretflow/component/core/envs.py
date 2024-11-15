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


from enum import auto, unique

from .common.types import AutoNameEnum
from .common.utils import to_bool


@unique
class Envs(AutoNameEnum):
    ENABLE_NN = auto()


def get_env(name: Envs, default=None):
    import os

    val = os.environ.get(name.value, default)
    return val if val != "" else default


def get_bool_env(name: Envs, default=False):
    return to_bool(get_env(name, default))
