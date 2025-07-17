# Copyright 2022 Ant Group Co., Ltd.
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


class SFException(Exception):
    def __init__(self, message: str, detail: dict = None):
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        result = self.message
        if self.detail:
            try:
                result += f"<{json.dumps(self.detail)}>"
            except Exception:
                result += f"<{self.detail}>"

        return result


class InvalidArgumentError(SFException):
    """
    Exception thrown if an argument does not match with the expected value.
    """

    pass


class InvalidStateError(SFException):
    """
    Exception thrown if the system is not in an appropriate state to
    execute a method.
    """

    pass


class NotSupportedError(SFException):
    """Raise when trigger a not supported operation."""

    pass


class YACLError(SFException):
    pass
