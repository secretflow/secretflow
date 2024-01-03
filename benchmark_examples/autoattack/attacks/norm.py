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

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow import reveal, tune
from secretflow.ml.nn.sl.attacks.norm_torch import NormAttack
from secretflow.tune.tune_config import RunConfig


def norm(config, *, alice, bob, app: TrainBase):
    label = reveal(app.train_label.partitions[app.device_y].data)
    norm_callback = NormAttack(alice if app.device_y == bob else bob, label)
    app.train(norm_callback)
    logging.warning(f"norm attack metrics = {norm_callback.get_attack_metrics()}")
    return norm_callback.get_attack_metrics()


def auto_norm(alice, bob, app: TrainBase):
    search_space = {
        'train_batch_size': tune.search.grid_search([64, 128]),
        'alice_fea_nums': tune.search.grid_search([i for i in range(8, 10)]),
    }
    trainable = tune.with_parameters(norm, alice=alice, bob=bob, app=app)
    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            storage_path=global_config.get_autoattack_path(),
            name=f"{type(app).__name__}_norm",
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="auc", mode="max").config
    logging.warning(f"the best acc = {result_config}")
