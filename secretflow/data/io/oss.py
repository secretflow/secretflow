# Copyright 2022 Ant Group Co., Ltd.
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

import base64
import os
from distutils.util import strtobool

import s3fs as s3

_S3_ENDPOINT = 'DATA_STORAGE_S3_ENDPOINT'
_S3_ACCESSKEYID = 'DATA_STORAGE_S3_ACCESSKEYID'
_S3_ACCESSSECRET = 'DATA_STORAGE_S3_SECRETKEY'
_S3_VIRTUALHOSTED = 'DATA_STORAGE_S3_VIRTUAL_HOSTED'

_SCHEME = 'oss://'


def s3fs():
    """Return a s3 filesystem instance."""
    endpoint = os.environ.get(_S3_ENDPOINT)
    ak = os.environ.get(_S3_ACCESSKEYID)
    sk = os.environ.get(_S3_ACCESSSECRET)

    assert endpoint is not None, f'{_S3_ENDPOINT} not set'
    assert ak is not None, f'{_S3_ACCESSKEYID} not set'
    assert sk is not None, f'{_S3_ACCESSSECRET} not set'

    addressing_style = 'path'
    try:
        if strtobool(os.environ.get(_S3_VIRTUALHOSTED)):
            addressing_style = 'virtual'
    except Exception:
        pass

    ak, sk = base64.b64decode(ak).decode("utf-8"), base64.b64decode(sk).decode("utf-8")
    if not endpoint.startswith('http'):
        endpoint = f'http://{endpoint}'

    return s3.S3FileSystem(
        anon=False,
        key=ak,
        secret=sk,
        client_kwargs={'endpoint_url': endpoint},
        config_kwargs={'s3': {'addressing_style': addressing_style}},
    )


def open(path, mode='rb'):
    """Open a oss object.

    Args:
        path: oss file path.
        mode: optional; open mode.

    Returns:
        A file-like object.
    """
    assert path.startswith(_SCHEME), f'Invalid path: {path}, should be oss://...'

    s3 = s3fs()
    return s3.open(path[len(_SCHEME) :], mode)
