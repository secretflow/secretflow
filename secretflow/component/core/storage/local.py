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

import os
import shutil
from io import BufferedIOBase
from pathlib import Path

from secretflow.spec.v1.data_pb2 import StorageConfig

from .base import Storage, StorageType


class LocalStorage(Storage):
    def __init__(self, config: StorageConfig) -> None:
        super().__init__(config)
        assert config.type == "local_fs"
        self._local_wd = config.local_fs.wd

    def get_full_path(self, remote_fn) -> str:
        full_path = os.path.join(self._local_wd, remote_fn)
        full_path = os.path.normpath(full_path)
        full_path = os.path.abspath(full_path)
        return full_path

    def get_type(self) -> StorageType:
        return StorageType.LOCAL_FS

    def get_size(self, path: str) -> int:
        full_path = self.get_full_path(path)
        return os.path.getsize(full_path)

    def get_reader(self, path: str) -> BufferedIOBase:
        return self.open(path, "rb")

    def get_writer(self, path: str) -> BufferedIOBase:
        return self.open(path, "wb")

    def open(self, path: str, mode: str) -> BufferedIOBase:
        full_path = self.get_full_path(path)
        if "w" in mode:
            Path(full_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            return open(full_path, mode)
        except FileNotFoundError:
            raise FileNotFoundError(f"{full_path} not found")
        except IsADirectoryError:
            raise IsADirectoryError(f"{full_path} is a directory")
        except Exception as e:
            raise e

    def remove(self, path: str) -> None:
        full_path = self.get_full_path(path)
        if not os.path.exists(full_path):
            raise ValueError(f"{full_path} not exist")
        return os.remove(full_path)

    def exists(self, path: str) -> bool:
        full_path = self.get_full_path(path)
        return os.path.exists(full_path)

    def mkdir(self, path: str) -> bool:
        Path(path).mkdir(parents=True, exist_ok=True)

    def download_file(self, remote_path: str, local_path: str) -> None:
        full_remote_path = self.get_full_path(remote_path)
        if not os.path.exists(full_remote_path):
            raise ValueError(f"file not exist {full_remote_path}")
        if not os.path.isfile(full_remote_path):
            raise ValueError(f"{full_remote_path} is not a file")
        if os.path.exists(local_path):
            if not os.path.isfile(local_path):
                raise ValueError(f"{local_path} is not a file")
            if os.path.samefile(full_remote_path, local_path):
                return
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(full_remote_path, local_path)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        if not os.path.exists(local_path):
            raise ValueError(f"{local_path} not exist.")
        if not os.path.isfile(local_path):
            raise ValueError(f"{local_path} is not a file")
        full_remote_path = self.get_full_path(remote_path)

        if os.path.exists(full_remote_path):
            if not os.path.isfile(full_remote_path):
                raise ValueError(f"{full_remote_path} is not a file")
            if os.path.samefile(full_remote_path, local_path):
                return
        Path(full_remote_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, full_remote_path)
