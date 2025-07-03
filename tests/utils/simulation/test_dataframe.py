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

import numpy as np
import pandas as pd
import pytest

from secretflow import reveal
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data import SPLIT_METHOD
from secretflow.utils.simulation.data.dataframe import create_df
from secretflow.utils.simulation.datasets import dataset
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def df():
    yield pd.read_csv(dataset('iris'))


@pytest.mark.mpc(parties=3)
def test_create_hdataframe_should_ok_when_input_dataframe(
    sf_production_setup_devices, df
):
    # WHEN
    hdf = create_df(
        df,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


@pytest.mark.mpc(parties=3)
def test_create_hdataframe_dirichlet_sample_method_should_ok_when_input_dataframe(
    sf_production_setup_devices, df
):
    from sklearn.preprocessing import LabelEncoder

    new_df = df.copy()
    label_encoder = LabelEncoder()
    new_df["class"] = label_encoder.fit_transform(new_df["class"])

    # WHEN
    hdf = create_df(
        new_df,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
        num_classes=4,
        alpha=100,
        random_state=1234,
        label_column="class",
        split_method=SPLIT_METHOD.DIRICHLET,
    )
    num_classes = reveal(hdf.partitions[sf_production_setup_devices.alice].data)[
        "class"
    ].values
    # fmt: off
    expect_classes = np.array(
        [
            0,0,0,0,0,0,1,1,2,0,
            1,0,2,2,2,2,0,1,1,2,
            1,0,2,0,0,1,1,1,2,1,
            1,0,0,2,0,0,0,1,2,0,
            2,0,1,1,1,2,2,0,2,1,
        ]
    )
    # fmt: on
    np.testing.assert_array_equal(expect_classes, num_classes)
    # THEN
    assert len(hdf.partitions) == 3


@pytest.mark.mpc(parties=3)
def test_create_hdataframe_label_skew_sample_method_should_ok_when_input_dataframe(
    sf_production_setup_devices, df
):
    from sklearn.preprocessing import LabelEncoder

    new_df = df.copy()
    label_encoder = LabelEncoder()
    new_df["class"] = label_encoder.fit_transform(new_df["class"])
    max_class_nums = 2
    # WHEN
    hdf = create_df(
        new_df,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
        num_classes=3,
        max_class_nums=max_class_nums,
        random_state=1234,
        label_column="class",
        split_method=SPLIT_METHOD.LABEL_SCREW,
    )

    # THEN
    assert len(hdf.partitions) == 3

    num_classes = np.unique(
        reveal(hdf.partitions[sf_production_setup_devices.alice].data)["class"].values
    )
    assert len(num_classes) == max_class_nums


@pytest.mark.mpc(parties=3)
def test_create_hdataframe_should_ok_when_input_file(df, sf_production_setup_devices):
    # WHEN
    hdf = create_df(
        dataset('iris'),
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


@pytest.mark.mpc
def test_create_hdataframe_should_ok_when_specify_indexes(
    sf_production_setup_devices, df
):
    # WHEN
    hdf = create_df(
        df,
        parts={
            sf_production_setup_devices.alice: (0, 50),
            sf_production_setup_devices.bob: (50, 150),
        },
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 2
    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


@pytest.mark.mpc
def test_create_hdataframe_should_ok_when_specify_percentage(
    sf_production_setup_devices, df
):
    # WHEN
    hdf = create_df(
        df,
        parts={
            sf_production_setup_devices.alice: 0.3,
            sf_production_setup_devices.bob: 0.7,
        },
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 2
    assert len(
        reveal(hdf.partitions[sf_production_setup_devices.alice].data)
    ) == 0.3 * len(df)
    assert len(
        reveal(hdf.partitions[sf_production_setup_devices.bob].data)
    ) == 0.7 * len(df)

    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


@pytest.mark.mpc(parties=3)
def test_create_vdataframe_should_ok(df, sf_production_setup_devices):
    # WHEN
    vdf = create_df(
        df,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=1,
    )

    # THEN
    assert len(vdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df,
        pd.concat([reveal(part.data) for part in vdf.partitions.values()], axis=1),
    )


@pytest.mark.mpc(parties=3)
def test_create_vdataframe_should_ok_when_input_callable(
    df, sf_production_setup_devices
):
    # WHEN
    hdf = create_df(
        lambda: pd.read_csv(dataset('iris')),
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=1,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df,
        pd.concat([reveal(part.data) for part in hdf.partitions.values()], axis=1),
    )


@pytest.mark.mpc
def test_create_vdataframe_should_error_when_illegal_source(
    sf_production_setup_devices,
):
    with pytest.raises(
        AssertionError, match='Callable source must return a pandas DataFrame'
    ):
        create_df(
            lambda: 1,
            parts=[sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )

    with pytest.raises(InvalidArgumentError, match='Unknown source type'):
        create_df(
            {},
            parts=[sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )


@pytest.mark.mpc
def test_create_vdataframe_should_error_when_illegal_parts(
    df, sf_production_setup_devices
):
    with pytest.raises(AssertionError, match='Parts should not be none or empty!'):
        create_df(df, parts=None)

    with pytest.raises(AssertionError, match='Parts shall be list like of PYUs'):
        create_df(df, parts=[1, 2])

    with pytest.raises(AssertionError, match='Keys of parts shall be PYU'):
        create_df(df, parts={1: 0.1})

    with pytest.raises(AssertionError, match='Sum of percentages shall be 1.0.'):
        create_df(
            df,
            parts={
                sf_production_setup_devices.alice: 0.1,
                sf_production_setup_devices.bob: 2.0,
            },
        )

    with pytest.raises(AssertionError, match='Not all dict values are percentages.'):
        create_df(
            df,
            parts={
                sf_production_setup_devices.alice: 0.1,
                sf_production_setup_devices.bob: '3',
            },
        )
