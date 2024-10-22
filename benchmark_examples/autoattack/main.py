# Copyright 2023 Ant Group Co., Ltd.
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

import gc
import logging
import os
import types
from typing import Callable, Dict, List

import click
import torch.cuda

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackBase, DefaultAttackCase
from benchmark_examples.autoattack.defenses.base import DefaultDefenseCase, DefenseBase
from benchmark_examples.autoattack.utils.sync_globals import sync_remote_globals
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.tune import ResultGrid
from secretflow.tune.tune_config import RunConfig, TuneConfig

try:
    import secretflow as sf

    v = sf.version
except ImportError as e:
    print(
        "Cannot find secretflow module, "
        "maybe try use: "
        "export PYTHONPATH='/path/to/secretflow'"
    )
    raise e

import ray

import benchmark_examples.autoattack.utils.dispatch as dispatch
import secretflow as sf
import secretflow.distributed as sfd
from benchmark_examples.autoattack import global_config
from secretflow import tune
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.utils.errors import NotSupportedError

_PARTIES = ['alice', 'bob']


def init_ray():
    ray.init(
        address=global_config.get_ray_cluster_address(),
        log_to_driver=True,
    )


def init_sf():
    debug_mode = global_config.is_debug_mode()
    sfd.set_distribution_mode(
        mode=DISTRIBUTION_MODE.SIMULATION if not debug_mode else DISTRIBUTION_MODE.DEBUG
    )
    sf.shutdown()
    address = global_config.get_ray_cluster_address()
    address = 'local' if address is None else address
    sf.init(
        _PARTIES,
        address=address,
        log_to_driver=True,
        omp_num_threads=os.cpu_count(),
        debug_mode=debug_mode,
    )
    alice = sf.PYU("alice")
    bob = sf.PYU("bob")
    return alice, bob


class AutoAttackResult:
    def __init__(self, results, best_results, metrics: Dict):
        self.results = results
        self.best_results: List = best_results
        self.metrics = metrics


def objective_trainning(
    config: Dict,
    *,
    app: ApplicationBase,
    attack: AttackBase | None = None,
    defense: DefenseBase | None = None,
    origin_global_configs: Dict | None = None,
) -> Dict[str, float]:
    """
    The target function for ml train, attack, defense.
    This function will be executed remote by tune when use auto mode.
    Returns:
        Dict[str, float]: A dict type metrics with app train history and attack metrics.
    """
    if origin_global_configs:
        sync_remote_globals(origin_global_configs)
    attack = DefaultAttackCase() if attack is None else attack
    defense = DefaultDefenseCase() if defense is None else defense
    with (
        app as app,
        attack as attack,
        defense as defense,
    ):
        # set the tune config into app, attack and defense
        app.set_config(config)
        attack.set_config(config)
        defense.set_config(config)
        # first add defense callbacks, then add attack callbacks.
        attack_callback = attack.build_attack_callback(app)
        defense_callback: Callback = defense.build_defense_callback(app, attack)
        callbacks = [defense_callback, attack_callback]
        callbacks = [v for v in callbacks if v is not None]
        callbacks = None if len(callbacks) == 0 else callbacks
        train_metrics = app.train(callbacks=callbacks)
        get_metrics_params = tuple()
        attack_metrics_params = attack.attack_metrics_params()
        if attack_metrics_params is not None:
            preds = app.predict(callbacks=callbacks)
            get_metrics_params = (sf.reveal(preds), *attack_metrics_params)
        attack_metrics = (
            attack_callback.get_attack_metrics(*get_metrics_params)
            if attack_callback
            else {}
        )
        metrics = {}
        metrics.update(attack_metrics)
        # append the origin application train histor
        for name, v in train_metrics.items():
            metrics[f"app_{name}"] = v
        logging.warning(f"RESULT: {app} {attack} {defense} metrics = {metrics}")
        return metrics


def _dispatch_cls(
    dataset: str, model: str, attack: str | None, defense: str | None
) -> tuple[type(ApplicationBase), type(AttackBase) | None, type(DefenseBase) | None]:
    if defense and defense.replace("-", "_") == "no_defense":
        defense = None
    if attack and attack.replace("-", "_") == 'no_attack':
        attack = None
    app_cls = dispatch.dispatch_application(dataset, model)
    attack_cls = None
    defense_cls = None
    if attack:
        attack_cls = dispatch.dispatch_attack(attack)
    if defense:
        defense_cls = dispatch.dispatch_defense(defense)
    return app_cls, attack_cls, defense_cls


def _construct_search_space(
    app: ApplicationBase,
    attack: AttackBase | None,
    defense: DefenseBase | None,
):
    # get app + attack + defense search space
    search_space: Dict = app.search_space()
    if attack:
        search_space.update(attack.search_space())
    if defense:
        search_space.update(defense.search_space())
    # remove all None in search space
    search_space = {k: v for k, v in search_space.items() if v is not None}
    # record the total trail number.
    total_trial_nums = 1
    # update search space to tune.grid_search
    for k, v in search_space.items():
        if isinstance(v, list):
            search_space[k] = tune.search.grid_search(v)
            total_trial_nums *= len(v)
        elif v is not None:
            search_space[k] = v
            total_trial_nums *= len(list(v.values())[0])

    logging.warning(
        f"Search space (with total {total_trial_nums} trails) of "
        f"{app} {attack} {defense} is {search_space}"
    )
    return search_space


def _get_cluster_resources(
    app: ApplicationBase, attack: AttackBase | None, defense: DefenseBase | None
) -> List[Dict[str, float]] | List[List[Dict[str, float]]]:
    debug_mode = global_config.is_debug_mode()
    use_gpu = global_config.is_use_gpu()
    if not debug_mode and use_gpu:
        raise NotImplemented(
            "Does not support using GPU for trainning without debug_mode."
        )
    cluster_resources_pack = app.resources_consumption()
    if defense:
        cluster_resources_pack = defense.update_resources_consumptions(
            cluster_resources_pack, app, attack
        )
    if attack:
        cluster_resources_pack = attack.update_resources_consumptions(
            cluster_resources_pack, app
        )
    if debug_mode:
        cluster_resources = cluster_resources_pack.get_debug_resources()
    else:
        cluster_resources = cluster_resources_pack.get_all_sim_resources()
    if not global_config.is_use_gpu():
        cluster_resources = [cr.without_gpu() for cr in cluster_resources]
    else:
        cluster_resources = [
            cr.handle_gpu_mem(global_config.get_gpu_config())
            for cr in cluster_resources
        ]
    logging.info(f"The preprocessed cluster resource = {cluster_resources}")
    return cluster_resources


def _get_metrics(
    results: ResultGrid,
    app: ApplicationBase,
    attack: AttackBase | None,
    defense: DefenseBase | None,
) -> AutoAttackResult:
    metrics = {}
    # app metrics need a prefix app_
    metrics.update({f"app_{k}": v for k, v in app.tune_metrics().items()})
    if attack:
        metrics.update(attack.tune_metrics())
    if defense:
        metrics.update(defense.tune_metrics(metrics))
    print(f"metricccc = {metrics}")
    log_content = f"BEST RESULT for {app} {attack} {defense}: \n"
    best_results = []
    for metric_name, metric_mode in metrics.items():
        best_result = results.get_best_result(metric=metric_name, mode=metric_mode)
        log_content += (
            f"  best config (name: {metric_name}, mode: {metric_mode}) = {best_result.config}\n"
            f"  best metrics = {best_result.metrics},\n"
        )
        best_results.append(best_result)
    logging.warning(log_content)
    return AutoAttackResult(results, best_results, metrics)


def case_valid_check(
    dataset: str,
    model: str,
    attack: str | None,
    defense: str | None,
):
    app_cls, attack_cls, defense_cls = _dispatch_cls(dataset, model, attack, defense)
    app_impl: ApplicationBase = app_cls(alice=None, bob=None)
    attack_impl: AttackBase | None = None
    if attack_cls:
        attack_impl = attack_cls(alice=None, bob=None)
        if not attack_impl.check_app_valid(app_impl):
            # if attack not in app_impl.support_attacks():
            raise NotSupportedError(
                f"Attack {attack} not supported in app {app_impl}! "
                f"If not correct, check the implement of 'check_app_valid' in class {attack_cls}"
            )
    if defense_cls:
        defense_impl = defense_cls(alice=None, bob=None)
        if attack_impl and not defense_impl.check_attack_valid(attack_impl):
            raise NotSupportedError(
                f"Defense {defense} not supported in attack {attack}! "
                f"If not correct, check the implement of 'check_attack_valid' in class {defense_impl}"
            )
        if not defense_impl.check_app_valid(app_impl):
            raise NotSupportedError(
                f"Defense {defense} not supported in application {app_impl}! "
                f"If not correct, check the implement of 'check_app_valid' in class {defense_impl}"
            )


def run_case(
    dataset: str,
    model: str,
    attack: str | None,
    defense: str | None,
    enable_tune: bool = False,
    objective: Callable = objective_trainning,
):
    """
    Run a singal case with dataset, model, attack, defense.
    """
    case_valid_check(dataset, model, attack, defense)
    alice, bob = init_sf()
    app_cls, attack_cls, defense_cls = _dispatch_cls(dataset, model, attack, defense)
    app_impl: ApplicationBase = app_cls(alice=alice, bob=bob)
    attack_impl: AttackBase | None = (
        attack_cls(alice=alice, bob=bob) if attack_cls else None
    )
    defense_impl: DefenseBase | None = (
        defense_cls(alice=alice, bob=bob) if defense_cls else None
    )
    objective_name = f"{dataset}_{model}_{attack}_{defense}"
    # give ray tune a readable objective name.
    objective = types.FunctionType(objective.__code__, globals(), name=objective_name)
    try:
        if not enable_tune:
            if global_config.need_monitor():
                from benchmark_examples.autoattack.utils.monitor import (
                    monitor_resource_usage,
                )

                objective = monitor_resource_usage(objective)
            return objective(
                {},
                app=app_impl,
                attack=attack_impl,
                defense=defense_impl,
                origin_global_configs=None,
            )

        else:
            if global_config.is_debug_mode():
                init_ray()

            search_space = _construct_search_space(app_impl, attack_impl, defense_impl)
            cluster_resources = _get_cluster_resources(
                app_impl, attack_impl, defense_impl
            )
            objective = tune.with_parameters(
                objective,
                app=app_impl,
                attack=attack_impl,
                defense=defense_impl,
                origin_global_configs=global_config.get_self_globals(),
            )
            tuner = tune.Tuner(
                objective,
                tune_config=TuneConfig(max_concurrent_trials=1000),
                run_config=RunConfig(
                    storage_path=global_config.get_cur_experiment_result_path(),
                    name=f"{dataset}_{model}_{attack}_{defense}",
                ),
                cluster_resources=cluster_resources,
                param_space=search_space,
            )
            results = tuner.fit()
            return _get_metrics(results, app_impl, attack_impl, defense_impl)
    finally:
        if global_config.is_debug_mode():
            ray.shutdown()
        else:
            sf.shutdown()
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


@click.command(no_args_is_help=True)
@click.argument("dataset_name", type=click.STRING, required=True)
@click.argument("model_name", type=click.STRING, required=True)
@click.argument("attack_name", type=click.STRING, required=False, default=None)
@click.argument("defense_name", type=click.STRING, required=False, default=None)
@click.option(
    "--enable_tune",
    is_flag=True,
    default=None,
    required=False,
    help='Whether to run in auto mode.',
)
@click.option(
    "--simple",
    is_flag=True,
    default=None,
    required=False,
    help='Whether to use simple testing for easy debugging.',
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    required=False,
    default=None,
    help='Wheter to run secretflow on the debug mode.',
)
@click.option(
    "--datasets_path",
    type=click.STRING,
    required=False,
    default=None,
    help='Datasets load path, default to "~/.secretflow/datasets"',
)
@click.option(
    "--autoattack_storage_path",
    type=click.STRING,
    required=False,
    default=None,
    help='Autoattack results storage path, default to "~/.secretflow/datasets"',
)
@click.option(
    "--use_gpu",
    is_flag=True,
    required=False,
    default=None,
    help="Whether to use GPU, default to False",
)
@click.option(
    "--enable_monitor",
    is_flag=True,
    required=False,
    default=None,
    help="Whether to enable resource monitor, default to False",
)
@click.option(
    "--ray_cluster_address",
    type=click.STRING,
    required=False,
    default=None,
    help="The existing ray cluster address to connect.",
)
@click.option(
    "--random_seed",
    type=click.STRING,
    required=False,
    default=None,
    help="To achieve reproducible.",
)
@click.option(
    "--config",
    type=click.STRING,
    required=False,
    default=None,
    help="The config yaml files, you can put all configs here.",
)
def run(
    dataset_name: str,
    model_name: str,
    attack_name: str | None,
    defense_name: str | None,
    enable_tune: bool,
    simple: bool,
    debug_mode: bool,
    datasets_path: str | None,
    autoattack_storage_path: str | None,
    use_gpu: bool,
    enable_monitor: bool,
    ray_cluster_address: str | None,
    random_seed: int | None,
    config: str | None,
):
    """
    Run single case with dataset, model, attack and defense.\n
    ****** The command need at least 2 args (dataset, model):\n
    ****** [1]: the dataset name, like 'cifar10', 'criteo', 'bank', etc.\n
    ****** [2]: the model name, like 'dnn', 'deepfm', etc.\n
    ****** [3]: the attack name like 'norm', 'lia', etc., can be empty.\n
    ****** [4]: the defense name like 'grad', etc., can be empty.\n
    ****** --auto --config: if active auto mode (need a config file to know search spaces.).\n
    ****** example:\n
    ****** python benchmark_examples/autoattack/main.py bank dnn\n
    ****** python benchmark_examples/autoattack/main.py bank dnn norm\n
    ****** python benchmark_examples/autoattack/main.py ban dnn norm grad\n
    ****** python benchmark_examples/autoattack/main.py ban dnn norm --enable_tune --config="path/to/config.yamml"\n
    """
    global_config.init_globalconfig(
        datasets_path=datasets_path,
        autoattack_storage_path=autoattack_storage_path,
        simple=simple,
        use_gpu=use_gpu,
        debug_mode=debug_mode,
        enable_monitor=enable_monitor,
        ray_cluster_address=ray_cluster_address,
        random_seed=random_seed,
        config=config,
    )
    run_case(dataset_name, model_name, attack_name, defense_name, enable_tune)


if __name__ == '__main__':
    run()
