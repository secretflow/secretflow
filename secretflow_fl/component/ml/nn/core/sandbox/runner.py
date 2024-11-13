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

from .dynamic_sandbox import runtime_module_container, subscript_wrapper
from .static_sandbox import check_and_transform
from .whitelists.common_whitelist import builtins_whitelist

_subscript_wrapper_name = "__subscript_wrapper__"


def prepare_globals(apis={}):
    global_dict = {
        _subscript_wrapper_name: subscript_wrapper,
    }

    try:
        import tensorflow as tf

        from .whitelists import applications
        from .whitelists.tensorflow_whitelist import tensorflow_whitelist
        from .whitelists.tensorflow_wrapper import tensorflow_wrapper

        apps = runtime_module_container(
            applications, applications.app_whitelist, applications.app_wrapper
        )

        tf = runtime_module_container(tf, tensorflow_whitelist, tensorflow_wrapper)
        global_dict.update(
            {
                "tf": tf,
                "keras": tf.keras,
                "layers": tf.keras.layers,
                "Module": tf.Module,
                "Model": tf.keras.Model,
                "Layer": tf.keras.layers.Layer,
                "apps": apps,
            }
        )
    except ImportError:
        pass

    global_dict.update(apis)
    global_dict["__builtins__"] = builtins_whitelist
    return global_dict


def run_code(code, apis={}):
    code_trans = check_and_transform(code, subscript_wrapper=_subscript_wrapper_name)

    global_dict = prepare_globals(apis)
    exec(code_trans, global_dict)
    return global_dict
