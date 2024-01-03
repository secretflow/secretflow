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

import inspect
import multiprocessing
import os
from typing import Callable

import click

from benchmark_examples.autoattack import global_config

try:
    import secretflow as sf

    print(f"secretflow version = {sf.__version__}")
except ImportError as e:
    print(
        "Cannot find secretflow module, "
        "maybe try use: "
        "export PYTHONPATH='/path/to/secretflow'"
    )
    raise e

import multiprocess
import ray

import secretflow as sf
import secretflow.distributed as sfd
from benchmark_examples.autoattack.utils.dispatch import dispatch
from secretflow import PYU
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


def init_ray_for_auto():
    ray.init(
        num_cpus=multiprocessing.cpu_count(),
        resources={party: multiprocessing.cpu_count() for party in _PARTIES},
        log_to_driver=True,
    )


class SfDevices:
    alice: PYU
    bob: PYU
    carol: PYU

    def __init__(self):
        pass


def run_sf(debug_mode=True):
    devices = SfDevices()
    sfd.set_distribution_mode(
        mode=DISTRIBUTION_MODE.SIMULATION if not debug_mode else DISTRIBUTION_MODE.DEBUG
    )
    sf.shutdown()
    sf.init(
        _PARTIES,
        address="local",
        num_cpus=32,
        log_to_driver=True,
        omp_num_threads=multiprocess.cpu_count(),
        debug_mode=debug_mode,
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    return devices


def run_func_in_sf(devices, App, func: Callable):
    assert isinstance(func, Callable), f'need callable but got func {type(func)}'
    app_kwargs = init_kwargs(App.__init__, devices)
    app = App(**app_kwargs)
    func_kwargs = init_kwargs(func, devices)
    func_name = func.__name__
    if func_name != 'train':
        attack = func_name.lstrip('auto_')
        if attack not in app.support_attacks():
            raise NotSupportedError(
                f"Attack {attack} not supported in app {App.__name__}! "
                f"If not correct, check the implement of 'support_attacks' in class {App.__name__}"
            )
    func_kwargs['app'] = app
    func(**func_kwargs)


def init_kwargs(func, devices):
    sig = inspect.signature(func)
    kwargs = {}
    if 'config' in sig.parameters:
        kwargs['config'] = {}
    if 'alice' in sig.parameters:
        kwargs['alice'] = devices.alice
    if 'bob' in sig.parameters:
        kwargs['bob'] = devices.bob
    if 'carol' in sig.parameters:
        kwargs['carol'] = devices.carol
    return kwargs


def run_case(ds, md, at, debug_mode):
    App, target_func = dispatch(ds, md, at)
    if debug_mode and 'auto' in at:  # need init ray by self.
        init_ray_for_auto()
    devices = run_sf(debug_mode=debug_mode)
    run_func_in_sf(devices, App, target_func)
    return 'Success'


@click.command()
@click.argument("dataset_name", type=click.STRING, required=True)
@click.argument("model_name", type=click.STRING, required=True)
@click.argument("run_mode", type=click.STRING, required=True)
@click.option("--debug_mode", type=click.BOOL, required=False)
@click.option(
    "--datasets_path",
    type=click.STRING,
    required=False,
    default=os.path.join(os.path.expanduser('~'), '.secretflow/datasets'),
)
@click.option(
    "--autoattack_storage_path",
    type=click.STRING,
    required=False,
    default=os.path.join(os.path.expanduser('~'), '.secretflow/workspace'),
)
def run(
    dataset_name,
    model_name,
    run_mode,
    debug_mode,
    datasets_path,
    autoattack_storage_path,
):
    try:
        global_config.set_dataset_path(datasets_path)
        global_config.set_autoattack_path(autoattack_storage_path)
        if not debug_mode:
            debug_mode = False if "auto" in run_mode else True
        run_case(dataset_name, model_name, run_mode, debug_mode)
    except Exception as e:
        show_helps()
        raise e


if __name__ == '__main__':
    run()
