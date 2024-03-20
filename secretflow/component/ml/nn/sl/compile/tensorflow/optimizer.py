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

import json

import tensorflow as tf


def compile_optimizer(name: str, params_str: str, learning_rate: float):
    params = {}
    if params_str:
        params = json.loads(params_str)

    if learning_rate > 0:
        params["learning_rate"] = learning_rate
        params["is_legacy_optimizer"] = False

    config = {"class_name": str(name).strip(), "config": params}

    return config


def get_optimizer(config):
    return tf.keras.optimizers.get(config)
