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
import platform
import shutil
from io import BufferedIOBase
from pathlib import Path
from typing import Dict

from secretflow.error_system.exceptions import SFTrainingHyperparameterError
from secretflow.spec.v1.data_pb2 import StorageConfig

from .base import _StorageBase


class _LocalStorage(_StorageBase):
    def __init__(self, config: StorageConfig) -> None:
        super().__init__()
        assert config.type == "local_fs"
        self._local_wd = config.local_fs.wd

    def download_file(self, remote_fn, local_fn) -> None:
        """blocked download whole file into local_fn, overwrite if local_fn exist"""
        full_remote_fn = os.path.join(self._local_wd, remote_fn)
        if not os.path.exists(full_remote_fn):
            raise SFTrainingHyperparameterError.file_not_exist(argument=full_remote_fn)
        if not os.path.isfile(full_remote_fn):
            raise SFTrainingHyperparameterError.not_a_file(argument=full_remote_fn)
        if os.path.exists(local_fn):
            if not os.path.isfile(local_fn):
                raise SFTrainingHyperparameterError.file_not_exist(argument=local_fn)
            if os.path.samefile(full_remote_fn, local_fn):
                return
        Path(local_fn).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(full_remote_fn, local_fn)

    def upload_file(self, remote_fn, local_fn) -> None:
        """blocked upload_file whole file into remote_fn, overwrite if remote_fn exist"""
        if not os.path.exists(local_fn):
            raise SFTrainingHyperparameterError.file_not_exist(argument=local_fn)
        if not os.path.isfile(local_fn):
            raise SFTrainingHyperparameterError.not_a_file(argument=local_fn)
        full_remote_fn = os.path.join(self._local_wd, remote_fn)

        if os.path.exists(full_remote_fn):
            if not os.path.isfile(full_remote_fn):
                raise SFTrainingHyperparameterError.not_a_file(argument=full_remote_fn)
            if os.path.samefile(full_remote_fn, local_fn):
                return
        Path(full_remote_fn).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_fn, full_remote_fn)

    def get_reader(self, remote_fn) -> BufferedIOBase:
        full_remote_fn = os.path.join(self._local_wd, remote_fn)
        if not os.path.exists(full_remote_fn):
            raise SFTrainingHyperparameterError.file_not_exist(argument=full_remote_fn)
        if not os.path.isfile(full_remote_fn):
            raise SFTrainingHyperparameterError.not_a_file(argument=full_remote_fn)
        return open(full_remote_fn, "rb")

    def remove(self, remote_fn) -> None:
        full_remote_fn = os.path.join(self._local_wd, remote_fn)
        if not os.path.exists(full_remote_fn):
            raise SFTrainingHyperparameterError.file_not_exist(argument=full_remote_fn)
        return os.remove(full_remote_fn)

    def exists(self, remote_fn) -> bool:
        full_remote_fn = os.path.join(self._local_wd, remote_fn)
        return os.path.exists(full_remote_fn)

    def get_writer(self, remote_fn) -> BufferedIOBase:
        full_remote_fn = os.path.join(self._local_wd, remote_fn)
        Path(full_remote_fn).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(full_remote_fn) and not os.path.isfile(full_remote_fn):
            raise SFTrainingHyperparameterError(
                f"full_remote_fn [{full_remote_fn}] already exists and is not a file"
            )
        return open(full_remote_fn, "wb")

    def get_file_meta(self, remote_fn) -> Dict:
        full_remote_fn = os.path.join(self._local_wd, remote_fn)
        if not os.path.exists(full_remote_fn):
            raise SFTrainingHyperparameterError.file_not_exist(argument=full_remote_fn)
        ret = {
            "ctime": os.path.getctime(full_remote_fn),
            "mtime": os.path.getmtime(full_remote_fn),
            "size": os.path.getsize(full_remote_fn),
        }
        if platform.system() == 'Linux':
            ret["inode"] = os.stat(full_remote_fn).st_ino
        return ret
