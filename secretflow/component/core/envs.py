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


from enum import Enum, auto, unique

from secretflow.utils.errors import NotSupportedError


class AutoNameEnum(Enum):
    def _generate_next_value_(name, *args):
        return name


@unique
class Envs(AutoNameEnum):
    ENABLE_NN = auto()


def get_env(name: Envs, default=None):
    import os

    val = os.environ.get(name.value, default)
    return val if val != "" else default


def get_bool_env(name: Envs, default=False):
    return to_bool(get_env(name, default))


TRUE_STRINGS = ["true", "yes", "y", "enable", "enabled", "1"]
FALSE_STRINGS = ["false", "no", "n", "disable", "disabled", "0"]


def to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in TRUE_STRINGS:
        return True
    if value.lower() in FALSE_STRINGS:
        return False
    raise NotSupportedError("Unknown boolean value", detail={"value": value})
