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

import logging

import ray.air
from ray import tune

from secretflow.utils.ray_compatibility import ray_version_less_than

TuneConfig = tune.TuneConfig


class RunConfig(ray.air.RunConfig):
    def __init__(self, **kwargs):
        if ray_version_less_than("2.5.0"):
            if "storage_path" in kwargs:
                logging.warning(
                    f"ray ({ray.__version__}) does not support storage_path in RunConfig, "
                    f"please install ray with version greater than 2.5.0"
                )
                kwargs.pop("storage_path")
                super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
