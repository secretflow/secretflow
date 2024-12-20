#!/usr/bin/env python3
# *_* coding: utf-8 *_*
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

from gc import callbacks
import os
import tempfile

import numpy as np
import tensorflow as tf
from torch import nn, optim
from torchmetrics import Accuracy, Precision

from secretflow.device import reveal
from examples.security.h_bd.agg_freqfed import FreqAggregator
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.fl.compress import COMPRESS_STRATEGY
from secretflow_fl.security.aggregation import SparsePlainAggregator
from secretflow_fl.utils.simulation.datasets_fl import load_cifar10_horiontal
from tests.ml.nn.fl.model_def import ConvNet_CIFAR10, SimpleCNN

_temp_dir = tempfile.mkdtemp()
import logging

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def _torch_model_with_cifar10(
    devices, model_def, data, label, strategy, backend, callbacks, **kwargs
):
    device_list = [devices.alice, devices.bob]
    server = devices.carol

    if strategy in COMPRESS_STRATEGY:
        aggregator = SparsePlainAggregator(server)
    else:
        aggregator = FreqAggregator(server)

    # spcify params
    dp_spent_step_freq = kwargs.get("dp_spent_step_freq", None)
    num_gpus = kwargs.get("num_gpus", 0)
    skip_bn = kwargs.get("skip_bn", False)
    fl_model = FLModel(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=aggregator,
        strategy=strategy,
        backend=backend,
        random_seed=1234,
        num_gpus=num_gpus,
        skip_bn=skip_bn,
    )
    history = fl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=1,
        batch_size=128,
        aggregate_freq=2,
        dp_spent_step_freq=dp_spent_step_freq,
    )
    result = fl_model.predict(data, batch_size=128)
    assert len(reveal(result[device_list[0]])) == 20000
    assert len(reveal(result[device_list[1]])) == 30000
    global_metric, _ = fl_model.evaluate(data, label, batch_size=128, random_seed=1234)
    print(history, global_metric)

    assert (
        global_metric[0].result().numpy()
        == history["global_history"]["val_multiclassaccuracy"][-1]
    )

    assert global_metric[0].result().numpy() > 0.1

    model_path_test = os.path.join(_temp_dir, "base_model")
    fl_model.save_model(model_path=model_path_test, is_test=True)
    model_path_dict = {
        devices.alice: os.path.join(_temp_dir, "alice_model"),
        devices.bob: os.path.join(_temp_dir, "bob_model"),
    }
    fl_model.save_model(model_path=model_path_dict, is_test=False)

    new_fed_model = FLModel(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=None,
        backend=backend,
        random_seed=1234,
        num_gpus=num_gpus,
        # callbacks=callbacks
    )
    new_fed_model.load_model(model_path=model_path_dict, is_test=False)
    new_fed_model.load_model(model_path=model_path_test, is_test=True)
    reload_metric, _ = new_fed_model.evaluate(
        data, label, batch_size=128, random_seed=1234
    )

    np.testing.assert_equal(
        [m.result().numpy() for m in global_metric],
        [m.result().numpy() for m in reload_metric],
    )


def test_torch_model(sf_simulation_setup_devices):
    (train_data, train_label), (test_data, test_label) = load_cifar10_horiontal(
        parts={
            sf_simulation_setup_devices.alice: 0.4,
            sf_simulation_setup_devices.bob: 0.6,
        },
        normalized_x=True,
        categorical_y=True,
    )

    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
    model_def = TorchModel(
        model_fn=SimpleCNN,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )
    alice = sf_simulation_setup_devices.alice
    # Test fed_avg_w with mnist
    logging.info('test_print' * 20)
    _torch_model_with_cifar10(
        devices=sf_simulation_setup_devices,
        model_def=model_def,
        data=train_data,
        label=train_label,
        strategy="fed_avg_w",
        backend="torch",
        callbacks=None,
    )
