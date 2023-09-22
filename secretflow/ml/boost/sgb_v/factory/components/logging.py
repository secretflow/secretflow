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
import time
from dataclasses import dataclass
from typing import Any

from secretflow.device import wait


# logging params will be set at each component
# these parameters are helpful for developers
@dataclass
class LoggingParams:
    """
    'verbose': bool. If write more logs.

        default: False.

    'wait_execution': bool. If use wait method to get actual execution time in logs.

        When enabled, operations will be executated immediately,
        and some parallel computation patterns will be linearized.

        Therefore, this operation will harm the performance,
        so the benchmark result will be slower than actual execution.

        When benchmarking the whole alogithm, keep this option to False.

        Only use this option,
        if we want to observe individual operation's execution time more accurately.

        default: False.
    """

    verbose: bool = False
    wait_execution: bool = False


logging_params_names = {'verbose', 'wait_execution'}


class LoggingTools:
    @staticmethod
    def logging_params_from_dict(params: dict) -> LoggingParams:
        verbose = bool(params.get('verbose', False))
        wait_execution = bool(params.get('wait_execution', False))
        return LoggingParams(verbose, wait_execution)

    @staticmethod
    def logging_params_write_dict(params: dict, setting: LoggingParams):
        params['verbose'] = setting.verbose
        params['wait_execution'] = setting.wait_execution

    @staticmethod
    def logging_start_templated_message(execution_name: str):
        start_time = time.perf_counter()
        logging.info("Begin execution: {}".format(execution_name))
        return start_time

    @staticmethod
    def wait_execution(execution: Any):
        wait(execution)

    @staticmethod
    def logging_end_templated_message(execution_name: str, start_time: float):
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info("End execution: {}, duration: {}".format(execution_name, duration))

    @staticmethod
    # for component with logging_params correctly setup only
    def enable_logging(fn):
        def logging_inner(self, *args, **kwargs):
            func_name = fn.__qualname__
            if self.logging_params.verbose:
                start_time = LoggingTools.logging_start_templated_message(func_name)
            response = fn(self, *args, **kwargs)
            if self.logging_params.wait_execution:
                LoggingTools.wait_execution(response)
            if self.logging_params.verbose:
                LoggingTools.logging_end_templated_message(func_name, start_time)
            return response

        return logging_inner
