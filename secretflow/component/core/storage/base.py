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

from secretflow.spec.v1.data_pb2 import StorageConfig

from ..common.types import BaseEnum


class StorageType(BaseEnum):
    LOCAL_FS = "local_fs"
    S3 = "s3"


class Storage(ABC):
    def __init__(self, config: StorageConfig) -> None:
        self.config = config

    @abstractmethod
    def get_type(self) -> StorageType:
        pass

    @abstractmethod
    def get_size(self, path: str) -> int:
        return

    @abstractmethod
    def get_full_path(self, path: str) -> str:
        pass

    @abstractmethod
    def get_reader(self, path: str) -> BufferedIOBase:
        pass

    @abstractmethod
    def get_writer(self, path: str) -> BufferedIOBase:
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> None:
        """blocked download whole file into local_path, overwrite if local_path exist"""
        pass

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """blocked upload_file whole file into remote_path, overwrite if remote_path exist"""
        pass
