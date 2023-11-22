import logging

from secretflow import reveal, tune


def test_tune_simple_func(sf_simulation_setup_devices):
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


def test_tune_simple_func_with_sf(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    def my_func(config, _alice, _bob):
        a = _alice(lambda x: x)(config['a'])
        b = _bob(lambda x: x)(config['b'])
        score = reveal(_bob(lambda a, b: a + b)(a.to(_bob), b))
        logging.warning(f"score = {score}")
        return {'score': score}

    search_space = {
        'a': tune.grid_search([0, 1]),
        'b': tune.grid_search([5, 6]),
    }
    trainable = tune.with_parameters(my_func, _alice=alice, _bob=bob)
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="score", mode="max").config
    assert result_config is not None
    assert result_config['a'] == 1 and result_config['b'] == 6
