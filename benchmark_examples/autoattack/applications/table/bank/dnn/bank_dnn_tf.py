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
from typing import List, Optional, Union

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_dnn_tf import DnnBase, DnnFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.preprocessing import LabelEncoder, MinMaxScaler
from secretflow.utils.simulation.datasets import load_bank_marketing


class BankDnn(TrainBase):
    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        pass

    def __init__(self, config, alice, bob):
        self.hidden_size = 64
        super().__init__(config, alice, bob)

    def _prepare_data(self):
        data = load_bank_marketing(
            parts={self.alice: (0, 4), self.bob: (4, 16)}, axis=1
        )
        label = load_bank_marketing(parts={self.alice: (16, 17)}, axis=1)
        encoder = LabelEncoder()
        data['job'] = encoder.fit_transform(data['job'])
        data['marital'] = encoder.fit_transform(data['marital'])
        data['education'] = encoder.fit_transform(data['education'])
        data['default'] = encoder.fit_transform(data['default'])
        data['housing'] = encoder.fit_transform(data['housing'])
        data['loan'] = encoder.fit_transform(data['loan'])
        data['contact'] = encoder.fit_transform(data['contact'])
        data['poutcome'] = encoder.fit_transform(data['poutcome'])
        data['month'] = encoder.fit_transform(data['month'])
        label = encoder.fit_transform(label)
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        random_state = 1234
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    def create_base_model(self, input_dim, output_dim):
        # Create model
        def create_model():
            import tensorflow as tf

            model = DnnBase([100, output_dim], ["relu", "relu"])
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=["accuracy", tf.keras.metrics.AUC()],
            )
            return model

        return create_model

    def _create_base_model_alice(self):
        return self.create_base_model(4, self.hidden_size)

    def _create_base_model_bob(self):
        return self.create_base_model(12, self.hidden_size)

    def _create_fuse_model(self):
        input_dim = self.hidden_size
        party_nums = 2
        output_dim = 1

        def create_model():
            import tensorflow as tf

            model = DnnFuse(
                [input_dim for _ in range(party_nums)],
                dnn_units_size=[input_dim, output_dim],
                dnn_units_activation=['relu', 'sigmoid'],
            )
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=["accuracy", tf.keras.metrics.AUC()],
            )
            return model

        return create_model


def run(config, *, alice, bob, callbacks=None):
    bank_dnn = BankDnn(config, alice, bob)
    train_data, train_label, test_data, test_label = bank_dnn._prepare_data()
    base_model_dict = {
        alice: bank_dnn._create_base_model_alice(),
        bob: bank_dnn._create_base_model_bob(),
    }
    # Define DP operations
    train_batch_size = 128
    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=alice,
        model_fuse=bank_dnn._create_fuse_model(),
        # dp_strategy_dict=dp_strategy_dict,
    )
    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=10,
        batch_size=train_batch_size,
        shuffle=True,
        verbose=1,
        validation_freq=1,
        # dp_spent_step_freq=dp_spent_step_freq,
        callbacks=callbacks,
    )
    logging.warning(history)


def bank_dnn_train2(config, alice, bob):
    run(config, alice=alice, bob=bob)
