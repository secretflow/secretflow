import logging

from secretflow import tune
from secretflow.tune.tune_config import RunConfig
from tests.ml.nn.sl.attack.test_torch_fia import do_test_sl_and_fia


def test_attack_torch_fia(sf_simulation_setup_devices):
    search_space = {
        'attack_epochs': tune.search.choice([20, 60, 120]),
        'optim_lr': tune.search.loguniform(1e-5, 1e-1),
    }
    trainable = tune.with_parameters(
        do_test_sl_and_fia,
        alice=sf_simulation_setup_devices.alice,
        bob=sf_simulation_setup_devices.bob,
    )
    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            storage_path="/root/work/sf-autoattack/secretflow/ray_results",
            name="torch_fia",
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="mean_model_loss", mode="min").config
    logging.warning(f"result config = {result_config}")
