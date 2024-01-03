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

import importlib
import importlib.util
import pkgutil

__APP_PREFIX = 'benchmark_examples/autoattack/applications'
__ATTACK_PREFIX = 'benchmark_examples/autoattack/attacks'


def train(app):
    app.train()


def _find_subpackages(package_name):
    modules = pkgutil.iter_modules(path=[package_name], prefix=package_name + "/")
    subpackages = []

    for module in modules:
        if module.ispkg:
            subpackages.append(module.name)
    subpackages.reverse()
    return subpackages


def _find_applications():
    applications = {}  # {(dataset, model), complete path}
    for sub_pkgs in _find_subpackages(__APP_PREFIX):
        for datasets_pkg in _find_subpackages(sub_pkgs):
            for model_pkg in _find_subpackages(datasets_pkg):
                applications[
                    (
                        datasets_pkg[datasets_pkg.rindex('/') + 1 :],
                        model_pkg[model_pkg.rindex('/') + 1 :],
                    )
                ] = model_pkg.replace('/', '.')

    return applications


def _find_attacks():
    modules = pkgutil.iter_modules(path=[__ATTACK_PREFIX], prefix=__ATTACK_PREFIX + ".")
    attacks = []
    for module in modules:
        module = importlib.import_module(module.name.replace('/', '.'))
        if '__init__' not in module.__name__:
            attacks.append(module.__name__[module.__name__.rindex(".") + 1 :])
    return attacks


APPLICATIONS = _find_applications()
ATTACKS = _find_attacks()


def dispatch(dataset_name: str, model_name: str, func_name: str):
    if (dataset_name, model_name) not in APPLICATIONS:
        raise RuntimeError(
            f"Provided datasets:{dataset_name}, model:{model_name} seems not beeng implement, please check by yourself."
        )
    func_name = func_name.replace("-", "_")

    attack_name = func_name.lstrip('auto_') if 'auto' in func_name else func_name
    if attack_name not in ATTACKS and func_name != 'train':
        raise RuntimeError(
            f"Provided function:{func_name} seems not implemented, please check by yourself."
        )
    app_module = importlib.import_module(APPLICATIONS[(dataset_name, model_name)])
    if not hasattr(app_module, 'App'):
        raise ModuleNotFoundError(
            f"Cannot find implemention of 'App' in application {dataset_name}:{model_name} "
        )
    App = getattr(app_module, 'App')
    if func_name == 'train':
        func = train
    else:
        attack_module = importlib.import_module(__ATTACK_PREFIX.replace('/', '.'))
        if not hasattr(attack_module, func_name):
            raise ModuleNotFoundError(
                f"Cannot find implemention of {func_name} in attack modules."
            )
        func = getattr(attack_module, func_name)
    return App, func
