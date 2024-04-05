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
import tarfile
from pathlib import Path


def pack(model_path: str, tmpdir: str):
    """Pack the model in `model_path` to a tar file in `tmpdir`."""

    tmp = Path(tmpdir)
    model_package_path = tmp.joinpath("model_package.tar.gz")
    model_package_path.parent.mkdir(parents=True, exist_ok=True)
    model_package_path.touch()
    with tarfile.open(model_package_path, "w:gz") as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))

    return model_package_path


def unpack(filename, tmpdir: str):
    """Unpack the tar file of `filename` to a dir in `tmpdir` contains model files."""

    from ..compile.compile import build_model_paths

    tmpdir = Path(tmpdir)
    model_path = tmpdir.joinpath("model")
    with tarfile.open(filename, "r:*") as tar:
        for member in tar.getmembers():
            member.name
        tar.extractall(model_path, filter="data")

    config = build_model_paths(model_path)
    if not config.loss_path.is_dir():
        config.loss_path = None
    if not config.server_base_path.is_dir():
        config.server_base_path = None

    return config
