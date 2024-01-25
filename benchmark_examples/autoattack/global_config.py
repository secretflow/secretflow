# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

_DATASETS_PATH = os.path.join(os.path.expanduser('~'), '.secretflow/datasets')
_AUTOATTACK_PATH = os.path.join(os.path.expanduser('~'), '.secretflow/workspace')


def set_dataset_path(dataset_path: str):
    global _DATASETS_PATH
    _DATASETS_PATH = dataset_path


def get_dataset_path() -> str:
    global _DATASETS_PATH
    return _DATASETS_PATH


def set_autoattack_path(autoattack_path: str):
    global _AUTOATTACK_PATH
    _AUTOATTACK_PATH = autoattack_path


def get_autoattack_path() -> str:
    global _AUTOATTACK_PATH
    return _AUTOATTACK_PATH
