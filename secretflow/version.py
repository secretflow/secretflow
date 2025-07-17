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

__version__ = "1.14.0.dev$$DATE$$"
__commit_id__ = "$$COMMIT_ID$$"
__docker_version__ = "$$DOCKER_VERSION$$"
__build_time__ = "$$BUILD_TIME$$"
__mode__ = "lite"


def is_lite_version():
    return __mode__ == "lite"


def build_message():
    msg = []
    msg.append(f"Secretflow {__mode__} {__version__}")

    if "$$" not in __commit_id__:
        msg.append(f"Build time ({__build_time__}) with commit id: {__commit_id__}")
    else:
        msg.append(f"Build time ({__build_time__})")

    if "$$" not in __docker_version__:
        msg.append(f"Distribution inside docker: {__docker_version__}")

    return "\n".join(msg)
