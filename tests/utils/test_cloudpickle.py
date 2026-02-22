# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import tempfile


def test_dumps_should_ok():
    code_block_1 = """
    import numpy as np


    def avg(data):
    return np.average(data, axis=1)

    from secretflow.utils.cloudpickle import code_position_independent_dumps as dumps
    print(dumps(avg))
    """

    code_block_2 = """
    import numpy as np


    def foo():
    pass

    def avg(data):
    return np.average(data, axis=1)

    from secretflow.utils.cloudpickle import code_position_independent_dumps as dumps
    print(dumps(avg))
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/1.py", "w") as f1:
            f1.write(code_block_1)

        with open(f"{tmp_dir}/2.py", "w") as f2:
            f2.write(code_block_2)

        p1 = subprocess.run(f"python {tmp_dir}/1.py", capture_output=True, shell=True)
        p2 = subprocess.run(f"python {tmp_dir}/2.py", capture_output=True, shell=True)
        assert p1.stdout == p2.stdout
