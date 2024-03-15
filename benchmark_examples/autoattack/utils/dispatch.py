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
from typing import Dict, List, Tuple

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackCase

__APP_PREFIX = 'benchmark_examples/autoattack/applications'
__ATTACK_PREFIX = 'benchmark_examples/autoattack/attacks'


def _find_subpackages(package_name):
    """find subpackages from the input prefix package name like 'x/xxx/xxx'"""
    modules = pkgutil.iter_modules(path=[package_name], prefix=package_name + "/")
    subpackages = []

    for module in modules:
        if module.ispkg:
            subpackages.append(module.name)
    subpackages.reverse()
    return subpackages


def _find_applications() -> Dict[Tuple[str, str], str]:
    """Find all aplications."""
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


def _find_attacks() -> List[str]:
    """Find all attacks."""
    module = importlib.import_module(__ATTACK_PREFIX.replace('/', '.'))
    attacks = [att for att in dir(module) if not att.startswith("__") and att != 'base']
    return attacks


APPLICATIONS: Dict[Tuple[str, str], str] = _find_applications()
ATTACKS: List[str] = _find_attacks()


def dispatch_application(dataset: str, model: str) -> type(ApplicationBase):
    """
    Dispatch an application class from the given dataset name and model name.
    Args:
        dataset (str): dataset name.
        model (str): model name.

    Returns:
        The class of the application.
    """
    if (dataset, model) not in APPLICATIONS:
        raise RuntimeError(
            f"Provided {dataset} - {model} seems not beeng implemented."
            f"The available applications are {list(APPLICATIONS.keys())}"
        )
    app_module = importlib.import_module(APPLICATIONS[(dataset, model)])
    if not hasattr(app_module, 'App'):
        raise ModuleNotFoundError(
            f"Cannot find implemention of 'App' in application {dataset}:{model}."
            f"Please make sure the Application is named as 'App' in __init__.py."
        )
    App = getattr(app_module, 'App')
    assert issubclass(App, ApplicationBase)
    return App


def dispatch_attack(attack: str) -> type(AttackCase):
    """
    Dispatch the attack class from the given attack name.
    Args:
        attack (str): attack name.

    Returns:
        The attack case class.
    """
    if attack not in ATTACKS:
        raise RuntimeError(
            f"Provided attack:{attack} seems not implemented."
            f"The avaliable attacks are {ATTACKS}"
        )
    attack_module = importlib.import_module(__ATTACK_PREFIX.replace('/', '.'))
    if not hasattr(attack_module, attack):
        raise ModuleNotFoundError(
            f"Cannot find implemention of {attack} in attack modules' __init__.py."
        )
    Attack = getattr(attack_module, attack)
    assert issubclass(Attack, AttackCase)
    return Attack
