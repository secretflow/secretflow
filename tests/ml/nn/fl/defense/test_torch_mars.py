# Copyright 2025 Ant Group Co., Ltd.
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

import os
import tempfile
import logging
import pytest

import numpy as np
import tensorflow as tf
from torch import nn, optim
from torchmetrics import Accuracy, Precision
from torchvision.datasets import FakeData

from secretflow.device import reveal
from examples.security.h_bd.agg_mars import MarsAggregator
from tests.ml.nn.fl.attack.fl_model_bd import FLModel_bd
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.fl.compress import COMPRESS_STRATEGY
from secretflow_fl.security.aggregation import SparsePlainAggregator
from secretflow_fl.utils.simulation.datasets_fl import load_cifar10_horiontal
from tests.ml.nn.fl.model_def import ConvNet_CIFAR10, SimpleCNN
from tests.ml.nn.fl.attack.backdoor_fl_torch import BackdoorAttack

# 临时目录，用于模型保存
_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

@pytest.fixture(autouse=True)
def cifar10_fake(monkeypatch):
    """
    在 CI 或开发环境中自动用 FakeData 替代真实 CIFAR-10 数据集下载，
    保持训练/测试样本数量与原始测试一致。
    """
    def _fake_loader(parts, normalized_x, categorical_y):
        # 构造假训练集：总样本 10000
        train_images = np.random.randint(0, 256, size=(10000, *INPUT_SHAPE), dtype=np.uint8)
        train_labels = np.random.randint(0, NUM_CLASSES, size=(10000,))
        # 构造假测试集：总样本 10000
        test_images = np.random.randint(0, 256, size=(10000, *INPUT_SHAPE), dtype=np.uint8)
        test_labels = np.random.randint(0, NUM_CLASSES, size=(10000,))
        # 根据 parts 划分 (返回与 load_cifar10_horiontal 相同结构)
        return (train_images, tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)), \
               (test_images,  tf.keras.utils.to_categorical(test_labels, NUM_CLASSES))

    monkeypatch.setattr(
        'secretflow_fl.utils.simulation.datasets_fl.load_cifar10_horiontal',
        _fake_loader
    )


def _torch_model_with_cifar10(
    devices,
    model_def,
    data,
    label,
    test_data,
    test_label,
    strategy,
    backend,
    callbacks,
    **kwargs
):
    device_list = [devices.alice, devices.bob, devices.carol, devices.davy]
    server = devices.carol

    if strategy in COMPRESS_STRATEGY:
        aggregator = SparsePlainAggregator(server)
    else:
        aggregator = MarsAggregator(server)

    dp_spent_step_freq = kwargs.get("dp_spent_step_freq", None)
    num_gpus = kwargs.get("num_gpus", 0)
    skip_bn = kwargs.get("skip_bn", False)
    fl_model = FLModel_bd(
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
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=128,
        aggregate_freq=1,
        dp_spent_step_freq=dp_spent_step_freq,
        callbacks=callbacks,
        attack_party=callbacks[0].attack_party,
        attack_epoch=1,
    )
    result = fl_model.predict(data, batch_size=128)
    assert len(reveal(result[device_list[0]])) == 10000
    assert len(reveal(result[device_list[1]])) == 10000
    assert len(reveal(result[device_list[2]])) == 20000
    assert len(reveal(result[device_list[3]])) == 10000

    global_metric, _ = fl_model.evaluate(
        test_data, test_label, batch_size=128, random_seed=1234
    )
    bd_metric, local_metric = fl_model.evaluate_bd(
        test_data,
        test_label,
        batch_size=128,
        random_seed=1234,
        attack_party=callbacks[0].attack_party,
        target_label=callbacks[0].target_label,
    )

    assert (
        global_metric[0].result().numpy()
        == history["global_history"]["val_multiclassaccuracy"][-1]
    )
    assert global_metric[0].result().numpy() > 0.05

    # 模型保存与加载测试
    model_path_test = os.path.join(_temp_dir, "base_model")
    fl_model.save_model(model_path=model_path_test, is_test=True)
    model_path_dict = {
        devices.alice: os.path.join(_temp_dir, "alice_model"),
        devices.bob: os.path.join(_temp_dir, "bob_model"),
        devices.carol: os.path.join(_temp_dir, "carol_model"),
        devices.davy: os.path.join(_temp_dir, "davy_model"),
    }
    fl_model.save_model(model_path=model_path_dict, is_test=False)

    new_fed_model = FLModel_bd(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=None,
        backend=backend,
        random_seed=1234,
        num_gpus=num_gpus,
    )
    new_fed_model.load_model(model_path=model_path_dict, is_test=False)
    new_fed_model.load_model(model_path=model_path_test, is_test=True)
    reload_metric, _ = new_fed_model.evaluate(
        data, label, batch_size=128, random_seed=1234
    )


def test_torch_model(sf_simulation_setup_devices):
    (train_data, train_label), (test_data, test_label) = load_cifar10_horiontal(
        parts={
            sf_simulation_setup_devices.alice: 0.2,
            sf_simulation_setup_devices.bob: 0.2,
            sf_simulation_setup_devices.carol: 0.4,
            sf_simulation_setup_devices.davy: 0.2,
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
    backdoor_attack = BackdoorAttack(
        attack_party=alice, poison_rate=0.01, target_label=1, eta=1.0, attack_epoch=1
    )
    _torch_model_with_cifar10(
        devices=sf_simulation_setup_devices,
        model_def=model_def,
        data=train_data,
        label=train_label,
        test_data=test_data,
        test_label=test_label,
        strategy="fed_avg_w",
        backend="torch",
        callbacks=[backdoor_attack],
    )
