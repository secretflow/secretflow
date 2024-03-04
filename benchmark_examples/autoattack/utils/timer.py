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


def readable_time(times: float) -> str:
    """get a readable string time like 2h38m"""
    times = int(times)
    readable = ""
    if times > 60 * 60 * 24:
        days = times // (60 * 60 * 24)
        readable += f"{days}d"
    if times > 60 * 60:
        hours = (times % (60 * 60 * 24)) // (60 * 60)
        readable += f"{hours}h"
    if times > 60:
        minutes = (times % (60 * 60)) // 60
        readable += f"{minutes}m"
    seconds = times % 60
    if readable == "":
        readable += f"{seconds}s"
    return readable
