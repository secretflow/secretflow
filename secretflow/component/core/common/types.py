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


import enum
import threading
import time
from dataclasses import dataclass
from typing import Callable

from secretflow.spec.v1.data_pb2 import DistData

Input = DistData


@dataclass
class Output:
    uri: str
    data: DistData = None  # type: ignore


@dataclass
class UnionSelection:
    name: str
    desc: str
    minor_min: int = 0
    minor_max: int = -1


class UnionGroup:
    """
    A group of union attrs.
    """

    def __init__(self) -> None:
        self._selected: str = ""

    def is_selected(self, v: str) -> bool:
        return self._selected == v

    def set_selected(self, v: str):
        self._selected = v

    def get_selected(self) -> str:
        return self._selected


class _Guard:
    def __init__(self, update_time_fn: Callable[[float], None]) -> None:
        self.update_time_fn = update_time_fn

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *_):
        elapsed_time = time.time() - self.start_time
        self.update_time_fn(elapsed_time)


class TimeTracer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.io_time = 0
        self.run_time = 0

    def _update_io_time(self, elapsed_time):
        with self.lock:
            self.io_time += elapsed_time

    def _update_run_time(self, elapsed_time):
        with self.lock:
            self.run_time += elapsed_time

    def trace_io(self):
        return _Guard(self._update_io_time)

    def trace_running(self):
        return _Guard(self._update_run_time)

    def report(self) -> dict[str, float]:
        return {"io_time": self.io_time, "run_time": self.run_time}


class AutoNameEnum(enum.Enum):
    def _generate_next_value_(name, *args):
        return name


class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(enum.Enum, metaclass=MetaEnum):
    def __repr__(self):
        return self.value

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return str(self) == str(other)
