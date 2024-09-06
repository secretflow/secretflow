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


from abc import ABC, abstractmethod
from io import BufferedIOBase
from typing import Dict


class _StorageBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def download_file(self, remote_fn, local_fn) -> None:
        """blocked download whole file into local_fn, overwrite if local_fn exist"""
        pass

    @abstractmethod
    def upload_file(self, remote_fn, local_fn) -> None:
        """blocked upload_file whole file into remote_fn, overwrite if remote_fn exist"""
        pass

    @abstractmethod
    def get_reader(self, remote_fn) -> BufferedIOBase:
        pass

    @abstractmethod
    def remove(self, remote_fn) -> None:
        pass

    @abstractmethod
    def exists(self, path) -> bool:
        pass

    @abstractmethod
    def get_writer(self, remote_fn) -> BufferedIOBase:
        pass

    @abstractmethod
    def get_file_meta(self, remote_fn) -> Dict:
        pass
