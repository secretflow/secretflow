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
import shutil
import tempfile
import uuid

import pytest

from secretflow import reveal
from secretflow_fl import tune
from secretflow_fl.tune.tune_config import RunConfig

_temp_dir = tempfile.mkdtemp()


def _do_tunning(devices, tune_config=None, run_config=None, cluster_resources=None):
    alice = devices.alice
    bob = devices.bob

    def my_func(config, _alice, _bob):
        a = _alice(lambda x: x)(config['a'])
        b = _bob(lambda x: x)(config['b'])
        score = reveal(_bob(lambda a, b: a + b)(a.to(_bob), b))
        return {'score': score}

    search_space = {
        'a': tune.grid_search([0, 1]),
        'b': tune.grid_search([5, 6]),
    }
    trainable = tune.with_parameters(my_func, _alice=alice, _bob=bob)
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config,
        cluster_resources=cluster_resources,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="score", mode="max").config
    assert result_config is not None
    assert result_config['a'] == 1 and result_config['b'] == 6


def test_tune_simple_func(sf_memory_setup_devices):
    def my_func(config):
        return {"score": config['a'] + config['b']}

    search_space = {
        'a': tune.grid_search([0, 1]),
        'b': tune.grid_search([5, 6]),
    }
    tuner = tune.Tuner(
        my_func,
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="score", mode="max").config
    assert result_config is not None
    assert result_config['a'] == 1 and result_config['b'] == 6


class TestMemTune:
    def test_mem_tune_with_default_resources(self, sf_memory_setup_devices):
        _do_tunning(sf_memory_setup_devices)

    def test_mem_tune_with_custom_resources(self, sf_memory_setup_devices):
        _do_tunning(
            sf_memory_setup_devices,
            cluster_resources=[{'CPU': 2}],
        )
        _do_tunning(
            sf_memory_setup_devices,
            cluster_resources={'CPU': 2},
        )

    def test_mem_tume_store_results(self, sf_memory_setup_devices):
        name = uuid.uuid4().hex
        _do_tunning(
            sf_memory_setup_devices,
            run_config=RunConfig(storage_path=_temp_dir, name=name),
        )
        assert os.path.exists(_temp_dir + f"/{name}")
        shutil.rmtree(_temp_dir + f"/{name}")


@pytest.mark.parametrize(
    "sf_simulation_setup_devices", [{"is_tune": True}], indirect=True
)
class TestSimTune:
    def test_sim_tune_with_default_resources(self, sf_simulation_setup_devices):
        _do_tunning(sf_simulation_setup_devices)

    def test_sim_tune_with_custom_resources(self, sf_simulation_setup_devices):
        _do_tunning(
            sf_simulation_setup_devices,
            cluster_resources=[{'alice': 1, 'CPU': 1}, {'bob': 1, 'CPU': 1}],
        )

    def test_sim_tume_store_results(self, sf_simulation_setup_devices):
        name = uuid.uuid4().hex
        _do_tunning(
            sf_simulation_setup_devices,
            run_config=RunConfig(storage_path=_temp_dir, name=name),
        )
        assert os.path.exists(_temp_dir + f"/{name}")
        shutil.rmtree(_temp_dir + f"/{name}")


class TestProdTune:
    # TODO: @xiaonan, when support prod mode, add a prod pytest fixture with 'sf_party_for_4pc'
    pass
