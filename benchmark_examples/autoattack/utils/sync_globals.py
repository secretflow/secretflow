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


def sync_remote_globals(origin_global_configs):
    """
    When a function is executed remotely, global variables are lost,
    so the global configs need to be synchronized.
    The function receives a configuration input to complete the synchronization process.
    Args:
        origin_global_configs: The original configs.
    """
    import benchmark_examples.autoattack.global_config as remote_global_config

    for k, v in origin_global_configs.items():
        setattr(remote_global_config, k, v)
