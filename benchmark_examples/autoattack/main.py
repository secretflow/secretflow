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

import logging
import os
from typing import List

import click

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackCase
from secretflow.tune.tune_config import RunConfig

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

import multiprocess
import ray

import benchmark_examples.autoattack.utils.dispatch as dispatch
import secretflow as sf
import secretflow.distributed as sfd
from benchmark_examples.autoattack import global_config
from secretflow import PYU, tune
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.utils.errors import NotSupportedError

_PARTIES = ['alice', 'bob']


def show_helps():
    print("****** Benchmark need at least 3 args:")
    print("****** [1]: the dataset name, like 'cifar10', 'criteo', 'bank', etc.")
    print("****** [2]: the model name, like 'dnn', 'deepfm', etc.")
    print(
        "****** [3]: the run mode, need 'train','predict','[attack]' or 'auto_[attack]'."
    )
    print("****** example:")
    print("****** python benchmark_examples/autoattack/main.py drive dnn train")
    print("****** python benchmark_examples/autoattack/main.py drive dnn auto-fia")


def init_ray():
    ray.init(
        log_to_driver=True,
    )


def init_sf():
    debug_mode = global_config.is_debug_mode()
    sfd.set_distribution_mode(
        mode=DISTRIBUTION_MODE.SIMULATION if not debug_mode else DISTRIBUTION_MODE.DEBUG
    )
    sf.shutdown()
    sf.init(
        _PARTIES,
        address="local",
        num_cpus=global_config.get_total_num_cpus(),
        num_gpus=(
            global_config.get_total_num_gpus() if global_config.is_use_gpu() else None
        ),
        log_to_driver=True,
        omp_num_threads=multiprocess.cpu_count(),
        debug_mode=debug_mode,
    )

    alice = sf.PYU("alice")
    bob = sf.PYU("bob")
    return alice, bob


class AutoAttackResult:
    def __init__(self, results, best_results, metric_names, metric_modes):
        self.results = results
        self.best_results: List = best_results
        self.metric_names: List = metric_names
        self.metric_modes: List = metric_modes


def do_train(dataset: str, model: str, alice: PYU, bob: PYU):
    App = dispatch.dispatch_application(dataset, model)
    app: ApplicationBase = App({}, alice, bob)
    app.prepare_data()
    app.train()


def do_attack(dataset: str, model: str, attack: str, alice: PYU, bob: PYU):
    App = dispatch.dispatch_application(dataset, model)
    Attack = dispatch.dispatch_attack(attack)
    if attack not in App({}, alice, bob).support_attacks():
        raise NotSupportedError(
            f"Attack {attack} not supported in app {App.__name__}! "
            f"If not correct, check the implement of 'support_attacks' in class {App.__name__}"
        )
    attack_case: AttackCase = Attack(alice, bob, App)
    attack_case.attack({})


def do_autoattack(dataset: str, model: str, attack: str, alice: PYU, bob: PYU):
    App = dispatch.dispatch_application(dataset, model)
    Attack = dispatch.dispatch_attack(attack)
    if attack not in App({}, alice, bob).support_attacks():
        raise NotSupportedError(
            f"Attack {attack} not supported in app {App.__name__}! "
            f"If not correct, check the implement of 'support_attacks' in class {App.__name__}"
        )
    attack_case: AttackCase = Attack(alice, bob, App, global_config.get_self_globals())
    search_space = attack_case.search_space()
    metric_names = attack_case.metric_name()
    metric_modes = attack_case.metric_mode()
    metric_names = (
        [metric_names] if not isinstance(metric_names, list) else metric_names
    )
    metric_modes = (
        [metric_modes] if not isinstance(metric_modes, list) else metric_modes
    )
    assert len(metric_names) == len(metric_modes)
    cluster_resources = [{'alice': 1, 'CPU': 1}, {'bob': 1, 'CPU': 1}]
    if global_config.is_use_gpu():
        cluster_resources = [
            {'alice': 1, 'CPU': 1, 'GPU': 0.5},
            {'bob': 1, 'CPU': 1, 'GPU': 0.5},
        ]
    tuner = tune.Tuner(
        attack_case.attack,
        run_config=RunConfig(
            storage_path=global_config.get_cur_experiment_result_path(),
            name=f"{dataset}_{model}_{attack}",
        ),
        cluster_resources=cluster_resources,
        param_space=search_space,
    )
    results = tuner.fit()
    log_content = ""
    best_results = []
    for metric_name, metric_mode in zip(metric_names, metric_modes):
        best_result = results.get_best_result(metric=metric_name, mode=metric_mode)
        log_content += f"RESULT: {dataset}_{model}_{attack} attack {metric_name}'s best config(mode={metric_mode}) = {best_result.config}, "
        f"best metrics = {best_result.metrics},\n"
        best_results.append(best_result)
    logging.warning(log_content)
    return AutoAttackResult(results, best_results, metric_names, metric_modes)


def run_case(dataset: str, model: str, attack: str):
    """
    Run a singal case with dataset, model and attack.
    """
    alice, bob = init_sf()
    if 'auto' in attack:
        init_ray()
        try:
            attack = attack.lstrip('auto_')
            return do_autoattack(dataset, model, attack, alice, bob)
        finally:
            ray.shutdown()
    elif attack == 'train':
        return do_train(dataset, model, alice, bob)
    else:
        return do_attack(dataset, model, attack, alice, bob)


@click.command()
@click.argument("dataset_name", type=click.STRING, required=True)
@click.argument("model_name", type=click.STRING, required=True)
@click.argument("run_mode", type=click.STRING, required=True)
@click.option("--simple", is_flag=True, default=None, help='whether use simple test.')
@click.option("--debug_mode", type=click.BOOL, required=False)
@click.option(
    "--datasets_path",
    type=click.STRING,
    required=False,
    default=None,
)
@click.option(
    "--autoattack_storage_path",
    type=click.STRING,
    required=False,
    default=os.path.join(os.path.expanduser('~'), '.secretflow/workspace'),
)
@click.option("--use_gpu", is_flag=True, required=False, default=False)
def run(
    dataset_name,
    model_name,
    run_mode,
    simple,
    debug_mode,
    datasets_path,
    autoattack_storage_path,
    use_gpu,
):
    try:
        global_config.init_globalconfig(
            datasets_path=datasets_path,
            autoattack_storage_path=autoattack_storage_path,
            simple=simple,
            use_gpu=use_gpu,
            debug_mode=debug_mode,
        )
        run_case(dataset_name, model_name, run_mode)
    except Exception as e:
        show_helps()
        raise e


if __name__ == '__main__':
    run()
