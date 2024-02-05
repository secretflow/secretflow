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

import inspect
from pathlib import Path
from typing import Callable

import tensorflow as tf

from secretflow.component.ml.nn.core.sandbox.runner import run_code


def _compile_func(func: Callable):
    class _CustomLoss(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=None, dtype=tf.float32),
                tf.TensorSpec(shape=None, dtype=tf.float32),
            ]
        )
        def __call__(self, y_true, y_pred):
            return func(y_true, y_pred)

    loss = _CustomLoss()
    return loss


def _compile_class(loss_obj: Callable):
    assert isinstance(loss_obj, tf.Module)
    return loss_obj


def build_apis(loss_path: Path):
    def _compile(func: Callable):
        loss = None
        if inspect.isfunction(func):
            loss = _compile_func(func)
        if isinstance(func, tf.Module):
            loss = _compile_class(func)

        assert isinstance(loss, tf.Module)
        tf.saved_model.save(loss, loss_path)

    return _compile


def compile_loss(builtin: str = None, custom_code: str = None, loss_path: Path = None):
    builtin = str(builtin).strip() if builtin is not None else builtin
    if builtin:
        return builtin

    assert custom_code, f"Loss function can not be empty."

    compile = build_apis(loss_path)

    apis = {
        "compile_loss": compile,
    }
    _ = run_code(custom_code, apis=apis)
    return None


def load_loss(loss_path: Path):
    loaded = tf.saved_model.load(loss_path)
    return loaded
