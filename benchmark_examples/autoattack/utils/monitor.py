# Copyright 2024 Ant Group Co., Ltd.
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

import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict

import GPUtil  # pip insatll gputil
import psutil  # pip install psutil

from .timer import readable_time


@dataclass
class ResourceUsage:
    start_time: int
    end_time: int
    start_memory: float
    max_memory: float
    max_cpu: float
    max_gpu: Dict[int, float]

    def __str__(self):
        gpu_str = {k: f"{v:.0f}MB" for k, v in self.max_gpu.items()}
        return (
            f"use time: {readable_time(self.end_time - self.start_time)},"
            f" max_memory: {self.max_memory:.0f}MB, start_mem: {self.start_memory:.0f} max_cpu: {self.max_cpu:.0f}%,"
            f" max_gpu: {gpu_str}"
        )


class MonitorThread(threading.Thread):
    def __init__(self):
        super().__init__(name="MonitorThread")
        self.killed = False
        self.start_time = 0
        self.end_time = 0
        self.start_memory = 0
        self.max_memory = 0
        self.max_cpu = 0
        self.max_gpu = {}

    def run(self):
        self.start_time = time.time()
        self.start_memory = self.get_memory(psutil.Process())
        while True:
            if self.killed:
                break
            p = psutil.Process()
            memory = self.get_memory(p)
            cpu = p.cpu_percent()
            self.max_memory = max(self.max_memory, memory)
            self.max_cpu = max(self.max_cpu, cpu)
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_memory = gpu.memoryUsed
                self.max_gpu[gpu.id] = max(self.max_gpu.get(gpu.id, 0), gpu_memory)
            time.sleep(0.1)
        self.end_time = time.time()

    def get_memory(self, p) -> float:
        memory_info = p.memory_full_info()
        memory = memory_info.rss / 1024 / 1024
        return memory

    def get_resource_usage(self) -> ResourceUsage:
        return ResourceUsage(
            start_time=self.start_time,
            end_time=self.end_time,
            start_memory=self.start_memory,
            max_memory=self.max_memory,
            max_cpu=self.max_cpu,
            max_gpu=self.max_gpu,
        )


def monitor_resource_usage(func: Callable[..., Dict]):
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MonitorThread()
        monitor.start()
        results = func(*args, **kwargs)
        assert isinstance(results, dict)
        monitor.killed = True
        monitor.join()
        resource_usage = monitor.get_resource_usage()
        results['resource_usage'] = resource_usage
        print(f"resource usage of {func.__name__}: {resource_usage}")
        return results

    return wrapper


class Monitor:
    def __init__(self):
        self.monitor_thread: MonitorThread | None = None

    def start(self):
        self.monitor_thread = MonitorThread()
        self.monitor_thread.start()

    def stop(self) -> ResourceUsage:
        self.monitor_thread.killed = True
        self.monitor_thread.join()
        return self.monitor_thread.get_resource_usage()
