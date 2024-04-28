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

from benchmark_examples.autoattack.benchmark import Benchmark


def get_supported_scenes():
    """Get the table of supported scenes."""
    check_supported = Benchmark(
        enable_tune=False, enable_log=False, objective=lambda *args, **kwargs: None
    )
    check_supported.run()
    return check_supported.experiments.to_markdown()


if __name__ == '__main__':
    """Run this script to print the supported applications/attacks/defenses"""
    print(get_supported_scenes())
