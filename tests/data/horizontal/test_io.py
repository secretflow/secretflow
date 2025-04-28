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

import os
import tempfile

import pandas as pd

from secretflow import reveal
from secretflow.data import partition
from secretflow.data.horizontal import HDataFrame, read_csv


def cleartmp(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def test_read_csv_and_to_csv_should_ok(sf_production_setup_devices_ray):
    # GIVEN
    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()
    file_uris = {
        sf_production_setup_devices_ray.alice: path1,
        sf_production_setup_devices_ray.bob: path2,
    }
    df1 = pd.DataFrame(
        {"c1": ["A5", "A1", "A2", "A6", "A7", "A9"], "c2": [5, 1, 2, 6, 2, 4]}
    )

    df2 = pd.DataFrame({"c1": ["B3", "B1", "B9", "B4"], "c2": [3, 1, 9, 4]})

    df = HDataFrame(
        {
            sf_production_setup_devices_ray.alice: partition(
                sf_production_setup_devices_ray.alice(lambda df: df)(df1)
            ),
            sf_production_setup_devices_ray.bob: partition(
                sf_production_setup_devices_ray.bob(lambda df: df)(df2)
            ),
        }
    )

    # WHEN
    df.to_csv(file_uris, index=False)

    # THEN
    # Waiting a while for to_csv finish.
    import time

    time.sleep(5)
    actual_df = read_csv(file_uris)
    pd.testing.assert_frame_equal(
        reveal(actual_df.partitions[sf_production_setup_devices_ray.alice].data), df1
    )
    pd.testing.assert_frame_equal(
        reveal(actual_df.partitions[sf_production_setup_devices_ray.bob].data), df2
    )
    cleartmp([path1, path2])
