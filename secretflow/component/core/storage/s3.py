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
from typing import Dict

import s3fs
from botocore import exceptions as s3_exceptions

from secretflow.error_system.exceptions import (
    NotSupportedError,
    SFTrainingHyperparameterError,
)
from secretflow.spec.v1.data_pb2 import StorageConfig

from .base import _StorageBase


class _S3Storage(_StorageBase):
    def __init__(self, config: StorageConfig) -> None:
        super().__init__()
        assert config.type == "s3"
        config: StorageConfig.S3Config = config.s3
        self._prefix = config.prefix
        self._bucket = config.bucket

        if config.version == "":
            config.version = "s3v4"

        if config.version not in [
            "s3v4",
            "s3v2",
        ]:
            raise NotSupportedError.not_supported_version(
                f"Not support S3Storage version {config.version}"
            )
        if not config.endpoint.startswith(
            "https://"
        ) and not config.endpoint.startswith("http://"):
            raise SFTrainingHyperparameterError.wrong_ip_address(
                f"Please specify the scheme(http or https) of endpoint"
            )
        self._s3_client = s3fs.S3FileSystem(
            anon=False,
            key=config.access_key_id,
            secret=config.access_key_secret,
            client_kwargs={'endpoint_url': config.endpoint},
            config_kwargs={
                'signature_version': config.version,
                's3': {
                    "addressing_style": "virtual" if config.virtual_host else "path"
                },
            },
        )

        try:
            self._s3_client.ls(self._bucket, detail=False)
        except s3_exceptions.UnknownSignatureVersionError as e:
            logging.exception(f"config.version {config.version} not support by server")
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

    def _full_remote_fn(self, remote_fn):
        return f"s3://{os.path.join(self._bucket, self._prefix, remote_fn)}"

    def download_file(self, remote_fn, local_fn) -> None:
        """blocked download whole file into local_fn, overwrite if local_fn exist"""
        full_remote_fn = self._full_remote_fn(remote_fn)
        try:
            self._s3_client.download(full_remote_fn, local_fn)
        except Exception as e:
            self._log_s3_error(e, full_remote_fn)
            raise

    def remove(self, remote_fn):
        """delete remote file"""
        full_remote_fn = self._full_remote_fn(remote_fn)
        try:
            self._s3_client.rm(full_remote_fn)
        except Exception as e:
            self._log_s3_error(e, full_remote_fn)
            raise

    def exists(self, remote_fn) -> bool:
        """is remote file exists"""
        full_remote_fn = self._full_remote_fn(remote_fn)
        return self._s3_client.exists(full_remote_fn)

    def upload_file(self, remote_fn, local_fn) -> None:
        """blocked upload whole file into remote_fn, overwrite if remote_fn exist"""
        full_remote_fn = self._full_remote_fn(remote_fn)
        try:
            self._s3_client.upload(local_fn, full_remote_fn)
        except Exception as e:
            self._log_s3_error(e)
            raise

    def get_reader(self, remote_fn) -> BufferedIOBase:
        full_remote_fn = self._full_remote_fn(remote_fn)
        try:
            return self._s3_client.open(full_remote_fn, "rb")
        except Exception as e:
            self._log_s3_error(e, full_remote_fn)
            raise

    def get_writer(self, remote_fn) -> BufferedIOBase:
        full_remote_fn = self._full_remote_fn(remote_fn)
        try:
            return self._s3_client.open(full_remote_fn, "wb")
        except Exception as e:
            self._log_s3_error(e)
            raise

    def get_file_meta(self, remote_fn) -> Dict:
        full_remote_fn = self._full_remote_fn(remote_fn)
        try:
            meta = self._s3_client.stat(full_remote_fn)
            ret = {
                "LastModified": meta["LastModified"],
                "size": meta["size"],
                "ETag": meta["ETag"],
            }
            return ret
        except Exception as e:
            self._log_s3_error(e)
            raise
