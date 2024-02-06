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

from secretflow.component.component import CompEvalError
from secretflow.component.env_utils import Envs, get_bool_env


def enabled():
    # disabled by default
    enable_nn = get_bool_env(Envs.ENABLE_NN, False)
    if not enable_nn:
        return False

    try:
        import tensorflow as tf

        logging.info(f"tensorflow version: {tf.__version__}")
        # full version
        return True
    except ImportError:
        # lite version
        return False


def check_enabled_or_fail():
    if not enabled():
        raise CompEvalError(
            f"slnn component is not enabled, please make sure tensorflow is installed and set env {Envs.ENABLE_NN.value}=true."
        )
