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

import logging
import tempfile

import pytest

from secretflow import reveal
from secretflow_fl import tune
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.callbacks.tune.automl import AutoMLCallback
from secretflow_fl.security.privacy import DPStrategy, LabelDP
from secretflow_fl.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
from secretflow_fl.utils.simulation.datasets_fl import load_mnist
from tests.fl.ml.nn.sl.model_def import create_base_model, create_fuse_model

_temp_dir = tempfile.mkdtemp()

num_classes = 10
input_shape = (28, 28, 1)
hidden_size = 64


def train(config, *, alice, bob):
    global _temp_dir
    _temp_dir = reveal(alice(lambda: tempfile.mkdtemp())())
    num_samples = 1000  # 原来是10000
    (x_train, y_train), (_, _) = load_mnist(
        parts={
            alice: (0, num_samples),
            bob: (0, num_samples),
        },
        normalized_x=True,
        categorical_y=True,
    )
    # prepare model
    num_classes = 10

    input_shape = (28, 28, 1)
    # keras model
    base_model = create_base_model(input_shape, 64, output_num=1)
    base_model_dict = {
        alice: base_model,
        bob: base_model,
    }
    fuse_model = create_fuse_model(
        input_dim=hidden_size,
        input_num=1,
        party_nums=len(base_model_dict),
        output_dim=num_classes,
    )

    # Define DP operations
    gaussian_embedding_dp = GaussianEmbeddingDP(
        noise_multiplier=0.5,
        l2_norm_clip=1.0,
        batch_size=config["train_batch_size"],
        num_samples=num_samples,
        is_secure_generator=False,
    )
    dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
    label_dp = LabelDP(eps=64.0)
    dp_strategy_bob = DPStrategy(label_dp=label_dp)
    dp_strategy_dict = {
        alice: dp_strategy_alice,
        bob: dp_strategy_bob,
    }
    dp_spent_step_freq = 10

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        dp_strategy_dict=dp_strategy_dict,
        simulation=True,
        random_seed=1234,
        backend="tensorflow",
        strategy="split_nn",
        base_local_steps=dp_spent_step_freq,
        fuse_local_steps=1,
        bound_param=0.0,
        loss_thres=0.01,
        split_steps=1,
        max_fuse_local_steps=10,
        agg_method=None,
        compressor=None,
        pipeline_size=1,
    )

    automl = AutoMLCallback()
    history = sl_model.fit(
        x_train,
        y_train,
        validation_data=(x_train, y_train),
        epochs=2,
        batch_size=config["train_batch_size"],
        shuffle=False,
        random_seed=1234,
        dataset_builder=None,
        audit_log_dir=_temp_dir,
        audit_log_params={"save_format": "h5"},
        callbacks=[automl],
    )
    logging.warning(f"accuracy = {history['val_accuracy'][-1]}")
    return {"accuracy": history["val_accuracy"][-1]}


@pytest.mark.parametrize(
    "sf_simulation_setup_devices", [{"is_tune": True}], indirect=True
)
def test_automl(sf_simulation_setup_devices):
    devices = sf_simulation_setup_devices
    search_space = {
        "train_batch_size": tune.grid_search([32, 512]),
    }
    trainable = tune.with_parameters(train, alice=devices.alice, bob=devices.bob)
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="accuracy", mode="max").config
    assert result_config is not None
    assert result_config["train_batch_size"] == 32
