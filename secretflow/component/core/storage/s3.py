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

import logging
import os
from io import BufferedIOBase

import s3fs
from botocore import exceptions as s3_exceptions

from secretflow.spec.v1.data_pb2 import StorageConfig

from .base import Storage, StorageType


class S3Storage(Storage):
    """
    s3 storage, please refer to https://s3fs.readthedocs.io/en/latest/
    """

    def __init__(self, config: StorageConfig) -> None:
        super().__init__(config)
        assert config.type == "s3"
        s3_config: StorageConfig.S3Config = config.s3

        if s3_config.version == "":
            s3_config.version = "s3v4"
        if s3_config.version not in ["s3v4", "s3v2"]:
            raise ValueError(f"Not support s3 version {s3_config.version}")

        if not s3_config.endpoint.startswith(("https://", "http://")):
            raise ValueError(
                f"Please specify the scheme(http or https) of endpoint<{s3_config.endpoint}>"
            )

        self._prefix = s3_config.prefix
        self._bucket = s3_config.bucket
        self._s3_client = s3fs.S3FileSystem(
            anon=False,
            key=s3_config.access_key_id,
            secret=s3_config.access_key_secret,
            client_kwargs={"endpoint_url": s3_config.endpoint},
            config_kwargs={
                "signature_version": s3_config.version,
                "s3": {
                    "addressing_style": "virtual" if s3_config.virtual_host else "path"
                },
            },
        )

        try:
            self._s3_client.ls(self._bucket, detail=False)
        except s3_exceptions.UnknownSignatureVersionError as e:
            logging.exception(
                f"config.version {s3_config.version} not support by server"
            )
            raise
        except Exception as e:
            self._log_s3_error(e)
            raise

    def _log_s3_error(self, e: Exception, file_name: str = None) -> None:
        if isinstance(e, FileNotFoundError):
            if file_name:
                logging.exception(
                    f"The file {file_name} in bucket {self._bucket} does not exist"
                )
            else:
                logging.exception(f"The specified bucket {self._bucket} does not exist")
        elif isinstance(e, PermissionError):
            logging.exception("Access denied, Check your key and signing method")
        else:
            logging.exception("Unknown error")

    def get_full_path(self, path: str) -> str:
        return f"s3://{os.path.join(self._bucket, self._prefix, path)}"

    def get_type(self) -> StorageType:
        return StorageType.S3

    def get_size(self, path: str) -> int:
        full_path = self.get_full_path(path)
        try:
            info = self._s3_client.stat(full_path)
            return info["size"]
        except Exception as e:
            self._log_s3_error(e)
            raise

    def get_reader(self, path: str) -> BufferedIOBase:
        return self.open(path, "rb")

    def get_writer(self, path: str) -> BufferedIOBase:
        return self.open(path, "wb")

    def open(self, path: str, mode: str) -> BufferedIOBase:
        full_path = self.get_full_path(path)
        try:
            return self._s3_client.open(full_path, mode)
        except Exception as e:
            self._log_s3_error(e)
            raise

    def remove(self, path: str) -> None:
        full_path = self.get_full_path(path)
        try:
            self._s3_client.rm(full_path)
        except Exception as e:
            self._log_s3_error(e, full_path)
            raise

    def exists(self, path: str) -> bool:
        full_path = self.get_full_path(path)
        return self._s3_client.exists(full_path)

    def mkdir(self, path: str):
        full_path = self.get_full_path(path)
        try:
            self._s3_client.mkdir(full_path)
        except Exception as e:
            self._log_s3_error(e, full_path)
            raise

    def download_file(self, remote_path: str, local_path: str) -> None:
        full_remote_path = self.get_full_path(remote_path)
        try:
            self._s3_client.download(full_remote_path, local_path)
        except Exception as e:
            self._log_s3_error(e, full_remote_path)
            raise

    def upload_file(self, local_path: str, remote_path: str) -> None:
        full_remote_fn = self.get_full_path(remote_path)
        try:
            self._s3_client.upload(local_path, full_remote_fn)
        except Exception as e:
            self._log_s3_error(e)
            raise
