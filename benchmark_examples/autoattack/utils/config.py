# Copyright 2024 Ant Group Co., Ltd.
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

import sys
from typing import Dict

yaml = None
try:
    import yaml
except ImportError:
    raise ImportError(
        'PyYAML is required when set config files, try "pip insatll pyyaml" first.'
    ).with_traceback(sys.exc_info()[2])


def read_config(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def read_tune_config(file_path: str) -> Dict:
    config = read_config(file_path)
    assert 'tune' in config, f"'tune' is required in {file_path} config file."
    return config['tune']
