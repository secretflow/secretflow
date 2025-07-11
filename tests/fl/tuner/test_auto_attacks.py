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

import logging

import pytest

from secretflow_fl import tune
from tests.fl.ml.nn.sl.attack.test_torch_fia import do_test_sl_and_fia


@pytest.mark.parametrize(
    "sf_simulation_setup_devices", [{"is_tune": True}], indirect=True
)
def test_attack_torch_fia(sf_simulation_setup_devices):
    devices = sf_simulation_setup_devices
    search_space = {
        'attack_epochs': tune.search.choice([20, 60, 120]),
        'optim_lr': tune.search.loguniform(1e-5, 1e-1),
    }
    trainable = tune.with_parameters(
        do_test_sl_and_fia, alice=devices.alice, bob=devices.bob
    )
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="mean_model_loss", mode="min").config
    logging.warning(f"result config = {result_config}")
