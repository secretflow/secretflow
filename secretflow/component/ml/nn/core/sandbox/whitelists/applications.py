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

from secretflow.ml.nn.applications.sl_bst_tf import BSTBase, BSTFuse, BSTPlusBase
from secretflow.ml.nn.applications.sl_deep_fm import DeepFMbase, DeepFMfuse
from secretflow.ml.nn.applications.sl_dnn_tf import DnnBase, DnnFuse
from secretflow.ml.nn.applications.sl_mmoe_tf import MMoEBase, MMoEFuse

from .tensorflow_wrapper import ModelWrapper

__all__ = [
    "DnnBase",
    "DnnFuse",
    "BSTBase",
    "BSTPlusBase",
    "BSTFuse",
    "DeepFMbase",
    "DeepFMfuse",
    "MMoEBase",
    "MMoEFuse",
]

app_whitelist = {name: {} for name in __all__}

# replace the class with a wrapper class at runtime
app_wrapper = {
    globals()[name]: type(name, (ModelWrapper, globals()[name]), {}) for name in __all__
}
