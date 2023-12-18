# Copyright 2023 Ant Group Co., Ltd.
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


_RAY_VERSION_2_0_0_STR = "2.0.0"


def _compare_version_strings(version1, version2):
    """
    This utility function compares two version strings and returns
    True if version1 is greater, and False if they're equal, and
    False if version2 is greater.
    """
    v1_list = version1.split('.')
    v2_list = version2.split('.')
    len1 = len(v1_list)
    len2 = len(v2_list)

    for i in range(min(len1, len2)):
        if v1_list[i] == v2_list[i]:
            continue
        else:
            break

    return int(v1_list[i]) > int(v2_list[i])


def ray_version_less_than_2_0_0():
    """Whther the current ray version is less 2.0.0."""
    import ray

    return _compare_version_strings(_RAY_VERSION_2_0_0_STR, ray.__version__)


def ray_version_less_than(version: str):
    import ray

    return _compare_version_strings(version, ray.__version__)
