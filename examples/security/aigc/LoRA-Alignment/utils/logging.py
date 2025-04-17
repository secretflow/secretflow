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
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("Rich")

__all__ = ["Logger", "logging"]


class Logger:

    def __init__(self, logger=None):
        self.logger = logger

    def info(self, msg):
        self.logger.info(msg) if self.logger else print(msg)

    def error(self, msg):
        self.logger.error(msg) if self.logger else print(msg)

    def warn(self, msg):
        self.logger.warn(msg) if self.logger else print(msg)
