import os
import shutil
import tempfile
import uuid

from secretflow import reveal, tune
from secretflow.tune.tune_config import RunConfig

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


def test_tune_simple_func(sf_tune_memory_setup_devices):
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

    def test_mem_tune_with_default_resources(self, sf_tune_memory_setup_devices):
        _do_tunning(sf_tune_memory_setup_devices)

    def test_mem_tune_with_custom_resources(self, sf_tune_memory_setup_devices):
        _do_tunning(
            sf_tune_memory_setup_devices,
            cluster_resources=[{'alice': 1, 'CPU': 1}, {'bob': 1, 'CPU': 1}],
        )
        _do_tunning(
            sf_tune_memory_setup_devices,
            cluster_resources={'alice': 1, 'bob': 1, 'CPU': 2},
        )

    def test_mem_tume_store_results(self, sf_tune_memory_setup_devices):
        name = uuid.uuid4().hex
        _do_tunning(
            sf_tune_memory_setup_devices,
            run_config=RunConfig(storage_path=_temp_dir, name=name),
        )
        assert os.path.exists(_temp_dir + f"/{name}")
        shutil.rmtree(_temp_dir + f"/{name}")


class TestSimTune:
    def test_mem_tune_with_default_resources(self, sf_tune_simulation_setup_devices):
        _do_tunning(sf_tune_simulation_setup_devices)

    def test_mem_tune_with_custom_resources(self, sf_tune_simulation_setup_devices):
        _do_tunning(
            sf_tune_simulation_setup_devices,
            cluster_resources=[{'alice': 1, 'CPU': 1}, {'bob': 1, 'CPU': 1}],
        )

    def test_mem_tume_store_results(self, sf_tune_simulation_setup_devices):
        name = uuid.uuid4().hex
        _do_tunning(
            sf_tune_simulation_setup_devices,
            run_config=RunConfig(storage_path=_temp_dir, name=name),
        )
        assert os.path.exists(_temp_dir + f"/{name}")
        shutil.rmtree(_temp_dir + f"/{name}")


class TestProdTune:
    # TODO: @xiaonan, when support prod mode, add a prod pytest fixture with 'sf_party_for_4pc'
    pass
