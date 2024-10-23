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


from io import BufferedIOBase

from secretflow.spec.v1.data_pb2 import StorageConfig

from .base import _StorageBase
from .local import _LocalStorage
from .s3 import _S3Storage


def _make_impl(config: StorageConfig) -> _StorageBase:  # type: ignore
    s_type = config.type.lower()
    if s_type == "s3":
        return _S3Storage(config)
    elif s_type == "local_fs":
        return _LocalStorage(config)
    else:
        raise ValueError(f"unsupported StorageConfig type {config.type}")


class Storage:
    def __init__(self, config) -> None:
        # make sure Storage is always pickle-able
        # Only reference pb config in this class
        self._config = config

    def download_file(self, remote_fn, local_fn) -> None:
        _make_impl(self._config).download_file(remote_fn, local_fn)

    def upload_file(self, remote_fn, local_fn) -> None:
        _make_impl(self._config).upload_file(remote_fn, local_fn)

    def get_reader(self, remote_fn) -> BufferedIOBase:
        return _make_impl(self._config).get_reader(remote_fn)

    def get_writer(self, remote_fn) -> BufferedIOBase:
        return _make_impl(self._config).get_writer(remote_fn)

    def get_file_meta(self, remote_fn) -> dict:
        return _make_impl(self._config).get_file_meta(remote_fn)

    def remove(self, remote_fn) -> None:
        impl = _make_impl(self._config)
        return impl.remove(remote_fn)

    def exists(self, remote_fn) -> bool:
        impl = _make_impl(self._config)
        return impl.exists(remote_fn)
