# Copyright 2024 The RayFed Team
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


import threading


class FedRuntimeError(Exception):
    pass


def main_thread_assert():
    if threading.current_thread() is not threading.main_thread():
        raise FedRuntimeError("Dont use fed api outside main thread")


class FedRemoteError(Exception):
    def __init__(self, src_party: str, cause: Exception) -> None:
        self._src_party = src_party
        self._cause = cause

    def __str__(self):
        error_msg = f'FedRemoteError occurred at {self._src_party}'
        if self._cause is not None:
            error_msg += f" caused by {str(self._cause)}"
        return error_msg


class FedLocalError(Exception):
    def __init__(self, cause: Exception) -> None:
        self._cause = cause
        self.with_traceback(cause.__traceback__)

    def cause(self):
        return self._cause

    def __str__(self):
        return str(self._cause)
