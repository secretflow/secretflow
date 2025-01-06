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


from secretflow.spec.v1.data_pb2 import StorageConfig

from .base import Storage, StorageType
from .local import LocalStorage
from .s3 import S3Storage


def make_storage(config: StorageConfig) -> Storage:
    if config.type == StorageType.LOCAL_FS:
        return LocalStorage(config)
    elif config.type == StorageType.S3:
        return S3Storage(config)
    else:
        raise ValueError(f"unsupported storage type{config.type}")


__all__ = [
    "make_storage",
    "StorageType",
    "Storage",
    "S3Storage",
    "LocalStorage",
]
