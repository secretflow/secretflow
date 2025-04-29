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

import tempfile
from pathlib import Path

from secretflow.device import PYU, PYUObject


def send(src_model_package_path: PYUObject, src_pyu: PYU, dst_pyu: PYU) -> PYUObject:
    """Send compiled model in `src_model_package_path` from `src_pyu` to `dst_pyu`."""

    def _load_package_bytes(path: str):
        with open(path, "rb") as f:
            data = f.read()
            return data

    src_data = src_pyu(_load_package_bytes)(src_model_package_path)
    dst_data = src_data.to(dst_pyu)

    def _save_package_bytes(dst_data):
        tmpdir = Path(tempfile.mkdtemp())
        model_package_path = tmpdir.joinpath("model_package.tar.gz")
        with open(model_package_path, "wb") as f:
            f.write(dst_data)

        return model_package_path

    dst_model_package_path = dst_pyu(_save_package_bytes)(dst_data)
    return dst_model_package_path
