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
import pytest

from secretflow import reveal
from secretflow.utils.simulation.data import SPLIT_METHOD
from secretflow.utils.simulation.data.ndarray import create_ndarray
from secretflow.utils.simulation.datasets import dataset


def get_ndarray():
    npz = np.load(dataset('mnist'))
    x_test = npz["x_test"]
    y_test = npz["y_test"]
    return x_test, y_test


@pytest.mark.mpc(parties=3)
def test_create_horizontal_fedndarray_should_ok(sf_production_setup_devices):
    # WHEN
    x_test, y_test = get_ndarray()
    fed_data = create_ndarray(
        x_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
    )
    fed_label = create_ndarray(
        y_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
    )

    # THEN
    assert len(fed_data.partitions) == 3
    assert len(fed_label.partitions) == 3
    alice_data = reveal(fed_data.partitions[sf_production_setup_devices.alice])
    alice_label = reveal(fed_label.partitions[sf_production_setup_devices.alice])
    assert alice_data.shape == (3333, 28, 28)
    assert alice_label.shape == (3333,)


@pytest.mark.mpc(parties=3)
def test_create_vertical_fedndarray_should_ok(sf_production_setup_devices):
    # WHEN
    x_test, y_test = get_ndarray()
    fed_data = create_ndarray(
        x_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=2,
    )
    fed_label = create_ndarray(
        y_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=1,
        is_label=True,
    )

    # THEN

    alice_data = reveal(fed_data.partitions[sf_production_setup_devices.alice])
    alice_label = reveal(fed_label.partitions[sf_production_setup_devices.alice])
    assert alice_data.shape == (10000, 28, 9)
    assert alice_label.shape == (10000,)


@pytest.mark.mpc(parties=3)
def test_create_horizontal_fedndarray_dirichlet_should_ok(sf_production_setup_devices):
    # WHEN
    x_test, y_test = get_ndarray()
    fed_data = create_ndarray(
        x_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        num_classes=10,
        alpha=100,
        random_state=1234,
        target=y_test,
        split_method=SPLIT_METHOD.DIRICHLET,
        axis=0,
    )

    fed_label = create_ndarray(
        y_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        num_classes=10,
        alpha=100,
        random_state=1234,
        target=y_test,
        split_method=SPLIT_METHOD.DIRICHLET,
        axis=0,
    )

    # # THEN
    assert len(fed_data.partitions) == 3
    assert len(fed_label.partitions) == 3
    alice_data = reveal(fed_data.partitions[sf_production_setup_devices.alice])
    alice_label = reveal(fed_label.partitions[sf_production_setup_devices.alice])
    assert alice_data.shape == (3333, 28, 28)
    assert alice_label.shape == (3333,)


@pytest.mark.mpc(parties=3)
def test_create_horizontal_fedndarray_label_skew_should_ok(sf_production_setup_devices):
    # WHEN
    x_test, y_test = get_ndarray()
    max_class_nums = 5
    fed_data = create_ndarray(
        x_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        num_classes=10,
        random_state=1234,
        target=y_test,
        max_class_nums=max_class_nums,
        split_method=SPLIT_METHOD.LABEL_SCREW,
        axis=0,
    )

    fed_label = create_ndarray(
        y_test,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        num_classes=10,
        random_state=1234,
        target=y_test,
        max_class_nums=max_class_nums,
        split_method=SPLIT_METHOD.LABEL_SCREW,
        axis=0,
    )

    # # THEN
    assert len(fed_data.partitions) == 3
    assert len(fed_label.partitions) == 3
    alice_data = reveal(fed_data.partitions[sf_production_setup_devices.alice])
    alice_label = reveal(fed_label.partitions[sf_production_setup_devices.alice])
    assert alice_data.shape == (3333, 28, 28)
    assert alice_label.shape == (3333,)
