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

from pathlib import Path

import tensorflow as tf

from secretflow.component.ml.nn.core.sandbox.runner import run_code
from secretflow.component.ml.nn.core.sandbox.whitelists.tensorflow_wrapper import (
    ModelWrapper,
)


def _compile(
    server_fuse: tf.keras.Model,
    server_base: tf.keras.Model,
    client_base: tf.keras.Model,
    server_fuse_path: Path,
    server_base_path: Path,
    client_base_path: Path,
):
    # server models
    if server_base is not None:
        assert isinstance(server_base, ModelWrapper)
        tf.saved_model.save(server_base, server_base_path)

    assert server_fuse is not None and isinstance(server_fuse, ModelWrapper)
    tf.saved_model.save(server_fuse, server_fuse_path)

    assert client_base is not None and isinstance(client_base, ModelWrapper)
    tf.saved_model.save(client_base, client_base_path)


def build_apis(server_fuse_path: Path, server_base_path: Path, client_base_path: Path):
    """build apis for compile and fit model.
    The api called `fit` but only to compile:
    ```python
    fit(
        server_fuse=fuse_model,
        server_base=base_model1,
        client_base=base_model2,
    )
    ```
    """

    def _fit(server_fuse, server_base, client_base):
        return _compile(
            server_fuse,
            server_base,
            client_base,
            server_fuse_path,
            server_base_path,
            client_base_path,
        )

    return _fit


def compile_models(
    code: str, server_fuse_path: Path, server_base_path: Path, client_base_path: Path
):
    fit = build_apis(server_fuse_path, server_base_path, client_base_path)
    apis = {
        "fit": fit,
    }

    _ = run_code(code, apis)


def load_models(server_base_path: Path, server_fuse_path: Path, client_base_path: Path):
    server_base = None
    if server_base_path.is_dir() and any(server_base_path.iterdir()):
        server_base = tf.saved_model.load(server_base_path)

    server_fuse = tf.saved_model.load(server_fuse_path)
    client_base = tf.saved_model.load(client_base_path)

    return server_fuse, server_base, client_base
