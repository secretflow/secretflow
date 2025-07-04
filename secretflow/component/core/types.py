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


import threading
import time
from typing import Callable

from secretflow.device.device.pyu import PYU
from secretflow.device.driver import wait


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


class PathCleanUp:
    def __init__(self, paths: dict[str:str]):
        self.paths = paths

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

    def cleanup(self):
        import shutil

        clean_res = []
        for party, root_dir in self.paths.items():
            res = PYU(party)(lambda v: shutil.rmtree(v))(root_dir)
            clean_res.append(res)

        wait(clean_res)
