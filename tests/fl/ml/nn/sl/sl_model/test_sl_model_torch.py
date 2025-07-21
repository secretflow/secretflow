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

import os
import tempfile

import numpy as np
from torch import nn, optim
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.device import reveal
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.agglayer.agg_method import Average
from secretflow_fl.security.privacy import DPStrategy
from secretflow_fl.security.privacy.mechanism.label_dp import LabelDP
from secretflow_fl.security.privacy.mechanism.torch import GaussianEmbeddingDP
from secretflow_fl.utils.compressor import TopkSparse
from secretflow_fl.utils.simulation.datasets_fl import load_mnist

from ..model_def import (
    ConvNetBase,
    ConvNetFuse,
    ConvNetFuseAgglayer,
    ConvNetRegBase,
    ConvNetRegFuse,
)

_temp_dir = tempfile.mkdtemp()

num_classes = 10
input_shape = (28, 28, 1)
train_batch_size = 64
num_samples = 1000


def create_dataset_builder(
    batch_size=32,
):
    def dataset_builder(x):
        import pandas as pd
        import torch
        import torch.utils.data as torch_data

        x = [t.values if isinstance(t, pd.DataFrame) else t for t in x]
        x_copy = [torch.tensor(t.copy()) for t in x]

        data_set = torch_data.TensorDataset(*x_copy)
        dataloader = torch_data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def torch_model_with_mnist(
    devices,
    base_model_dict,
    device_y,
    model_fuse,
    data,
    label,
    strategy="split_nn",
    backend="torch",
    **kwargs
):
    # kwargs parsing
    dp_strategy_dict = kwargs.get("dp_strategy_dict", None)
    dataset_builder = kwargs.get("dataset_builder", None)
    callbacks = kwargs.get("callbacks", None)
    load_base_model_dict = kwargs.get("load_base_model_dict", base_model_dict)
    load_model_fuse = kwargs.get("load_model_fuse", model_fuse)

    base_local_steps = kwargs.get("base_local_steps", 1)
    fuse_local_steps = kwargs.get("fuse_local_steps", 1)
    bound_param = kwargs.get("bound_param", 0.0)

    loss_thres = kwargs.get("loss_thres", 0.01)
    split_steps = kwargs.get("split_steps", 1)
    max_fuse_local_steps = kwargs.get("max_fuse_local_steps", 10)

    agg_method = kwargs.get("agg_method", None)
    compressor = kwargs.get("compressor", None)
    pipeline_size = kwargs.get("pipeline_size", 1)

    atol = kwargs.get("atol", 0.02)

    party_shape = data.partition_shape()
    alice_length = party_shape[devices.alice][0]

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
        dp_strategy_dict=dp_strategy_dict,
        simulation=True,
        random_seed=1234,
        backend=backend,
        strategy=strategy,
        base_local_steps=base_local_steps,
        fuse_local_steps=fuse_local_steps,
        bound_param=bound_param,
        loss_thres=loss_thres,
        split_steps=split_steps,
        max_fuse_local_steps=max_fuse_local_steps,
        agg_method=agg_method,
        compressor=compressor,
        pipeline_size=pipeline_size,
    )
    history = sl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=2,
        batch_size=train_batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_builder,
        callbacks=callbacks,
    )
    global_metric = sl_model.evaluate(
        data,
        label,
        batch_size=128,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )

    # test history
    print(global_metric)
    print(history)
    assert np.isclose(
        global_metric["MulticlassAccuracy"],
        history["val_MulticlassAccuracy"][-1],
        atol=atol,
    )
    if pipeline_size <= 1:
        assert global_metric["MulticlassAccuracy"] > 0.7
    else:
        assert global_metric["MulticlassAccuracy"] > 0.5

    result = sl_model.predict(data, batch_size=128, verbose=1)
    reveal_result = []
    for rt in result:
        reveal_result.extend(reveal(rt))
    assert len(reveal_result) == alice_length
    base_model_path = os.path.join(_temp_dir, "base_model")
    fuse_model_path = os.path.join(_temp_dir, "fuse_model")
    sl_model.save_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
        is_test=True,
    )
    sl_model_load = SLModel(
        base_model_dict=load_base_model_dict,
        device_y=device_y,
        model_fuse=load_model_fuse,
        dp_strategy_dict=dp_strategy_dict,
        compressor=compressor,
        simulation=True,
        random_seed=1234,
        backend=backend,
        strategy=strategy,
        base_local_steps=base_local_steps,
        fuse_local_steps=fuse_local_steps,
        bound_param=bound_param,
    )
    sl_model_load.load_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
        is_test=True,
    )
    reload_metric = sl_model_load.evaluate(
        data,
        label,
        batch_size=128,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )
    assert np.isclose(
        global_metric["MulticlassAccuracy"],
        reload_metric["MulticlassAccuracy"],
        atol=atol,
    )


class TestSLModelTorch:
    def test_torch_model(self, sf_simulation_setup_devices):
        alice = sf_simulation_setup_devices.alice
        bob = sf_simulation_setup_devices.bob
        (_, _), (mnist_data, mnist_label) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: (0, num_samples),
                sf_simulation_setup_devices.bob: (0, num_samples),
            },
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )
        mnist_data = mnist_data.astype(np.float32)
        mnist_label = mnist_label.astype(np.float32)

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
        base_model = TorchModel(
            model_fn=ConvNetBase,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )

        fuse_model = TorchModel(
            model_fn=ConvNetFuse,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )
        base_model_dict = {
            alice: base_model,
            bob: base_model,
        }
        # Define DP operations
        gaussian_embedding_dp = GaussianEmbeddingDP(
            noise_multiplier=0.5,
            l2_norm_clip=1.0,
            batch_size=128,
            num_samples=num_samples,
            is_secure_generator=False,
        )
        dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
        label_dp = LabelDP(eps=64.0)
        dp_strategy_bob = DPStrategy(label_dp=label_dp)
        dp_strategy_dict = {
            sf_simulation_setup_devices.alice: dp_strategy_alice,
            sf_simulation_setup_devices.bob: dp_strategy_bob,
        }
        # Test sl with dp
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            dp_strategy_dict=dp_strategy_dict,
            strategy="split_nn",
            backend="torch",
            atol=0.04,
        )
        # test dataset builder
        print("test Dataset builder")
        dataset_buidler_dict = {
            sf_simulation_setup_devices.alice: create_dataset_builder(
                batch_size=train_batch_size,
            ),
            sf_simulation_setup_devices.bob: create_dataset_builder(
                batch_size=train_batch_size,
            ),
        }
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            strategy="split_nn",
            backend="torch",
            dataset_builder=dataset_buidler_dict,
        )

        # test compressor
        print("test TopkSparse")
        top_k_compressor = TopkSparse(0.5)
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            strategy="split_nn",
            backend="torch",
            compressor=top_k_compressor,
        )

        # pipeline
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            strategy="pipeline",
            backend="torch",
            pipeline_size=2,
        )

    def test_single_feature_model_agg_layer(self, sf_simulation_setup_devices):
        alice = sf_simulation_setup_devices.alice
        bob = sf_simulation_setup_devices.bob
        (x_train, y_train), (_, _) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: (0, num_samples),
                sf_simulation_setup_devices.bob: (0, num_samples),
            },
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
        base_model = TorchModel(
            model_fn=ConvNetBase,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )

        fuse_model = TorchModel(
            model_fn=ConvNetFuseAgglayer,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )
        base_model_dict = {
            alice: base_model,
        }

        # agg layer
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=x_train,
            label=y_train,
            strategy="split_nn",
            backend="torch",
            agg_method=Average(),
        )

    def test_single_feature_model(self, sf_simulation_setup_devices):
        alice = sf_simulation_setup_devices.alice
        bob = sf_simulation_setup_devices.bob
        (_, _), (mnist_data, mnist_label) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: (0, num_samples),
                sf_simulation_setup_devices.bob: (0, num_samples),
            },
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )
        mnist_data = mnist_data.astype(np.float32)
        mnist_label = mnist_label.astype(np.float32)
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
        base_model = TorchModel(
            model_fn=ConvNetBase,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )

        fuse_model = TorchModel(
            model_fn=ConvNetFuseAgglayer,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average="micro"
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )
        base_model_dict = {
            alice: base_model,
        }

        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            strategy="split_nn",
            backend="torch",
        )

    def test_torch_model_custom_loss(self, sf_simulation_setup_devices):
        alice = sf_simulation_setup_devices.alice
        bob = sf_simulation_setup_devices.bob
        (_, _), (mnist_data, mnist_label) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: (0, num_samples),
                sf_simulation_setup_devices.bob: (0, num_samples),
            },
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )
        mnist_data = mnist_data.astype(np.float32)
        mnist_label = mnist_label.astype(np.float32)

        fuse_model = ConvNetRegFuse
        base_model_dict = {
            alice: ConvNetRegBase,
            bob: ConvNetRegBase,
        }
        dataset_buidler_dict = {
            sf_simulation_setup_devices.alice: create_dataset_builder(
                batch_size=train_batch_size,
            ),
            sf_simulation_setup_devices.bob: create_dataset_builder(
                batch_size=train_batch_size,
            ),
        }

        # Define DP operations
        gaussian_embedding_dp = GaussianEmbeddingDP(
            noise_multiplier=0.5,
            l2_norm_clip=1.0,
            batch_size=128,
            num_samples=num_samples,
            is_secure_generator=False,
        )
        dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
        label_dp = LabelDP(eps=64.0)
        dp_strategy_bob = DPStrategy(label_dp=label_dp)
        dp_strategy_dict = {
            sf_simulation_setup_devices.alice: dp_strategy_alice,
            sf_simulation_setup_devices.bob: dp_strategy_bob,
        }
        # Test sl with dp
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            dp_strategy_dict=dp_strategy_dict,
            strategy="split_nn",
            backend="torch",
            atol=0.05,
        )
        # Test sl with dataset builder and compressor
        top_k_compressor = TopkSparse(0.5)
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            strategy="split_nn",
            backend="torch",
            compressor=top_k_compressor,
            dataset_builder=dataset_buidler_dict,
        )
        # Test sl with pipeline
        torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            device_y=bob,
            model_fuse=fuse_model,
            data=mnist_data,
            label=mnist_label,
            strategy="pipeline",
            pipeline_size=2,
            backend="torch",
            dataset_builder=dataset_buidler_dict,
        )
