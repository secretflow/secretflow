# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from secretflow import PYU
from secretflow.ml.nn.callbacks import Callback
from secretflow.ml.nn.core.torch import BaseModule, module


def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / (torch.pow((1 - probabilities), 1 / T) + tempered)

    else:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    return torch.mean(H)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=2, encode_dim=3):
        super(AutoEncoder, self).__init__()
        self.d = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim**2),
            nn.ReLU(),
            nn.Linear(encode_dim**2, input_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim**2),
            nn.ReLU(),
            nn.Linear(encode_dim**2, input_dim),
            nn.Softmax(dim=1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal(m.weight.data)

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        d_y = F.softmax(z, dim=1)
        d_y = sharpen(d_y, T=1.0)
        return self.decoder(d_y), d_y

    def load_model(self, model_name, target_device='cuda'):
        self.load_state_dict(torch.load(model_name, map_location=target_device))

    def save_model(self, model_name):
        torch.save(self.state_dict(), model_name)


class AutoEncoderTrainer:
    def __init__(
        self,
        model,
        exec_device,
        num_classes,
        epochs,
        batch_size,
        T,
        hyper_lambda,
        train_sample_size,
        test_sample_size,
        learning_rate,
    ):
        self.model = model
        self.exec_device = exec_device
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.T = T
        self.hyper_lambda = hyper_lambda
        self.learning_rate = learning_rate

    def train(self):
        train_y = torch.rand(self.train_sample_size, self.num_classes)
        train_y = sharpen(F.softmax(train_y, dim=1), T=self.T)
        train_y = train_y.to(self.exec_device)

        test_y = torch.rand(self.test_sample_size, self.num_classes)
        test_y = sharpen(F.softmax(test_y, dim=1), T=self.T)
        test_y = test_y.to(self.exec_device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            for batch_idx in range(0, self.train_sample_size, self.batch_size):
                batch_y = train_y[batch_idx : batch_idx + self.batch_size]

                label_y = torch.argmax(batch_y, dim=1)

                y_hat, y_dummy = self.model(batch_y)

                label_y_hat = torch.argmax(y_hat, dim=1)  # decoder(encoder(y))
                label_y_dummy = torch.argmax(y_dummy, dim=1)  # encoder(y)

                loss_e = entropy(y_dummy)  # for confustion
                loss_p = criterion(y_hat, batch_y)  # for decoder acc
                loss_n = criterion(y_dummy, batch_y)
                loss = 10 * loss_p - self.hyper_lambda * loss_e - loss_n

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_acc_p = torch.sum(label_y_hat == label_y) / float(len(label_y))
                train_acc_n = torch.sum(label_y_dummy != label_y) / float(len(label_y))
                train_loss = loss.item()

            test_label_y = torch.argmax(test_y, dim=1)
            test_y_hat, test_y_dummy = self.model(test_y)
            test_label_y_hat = torch.argmax(test_y_hat, dim=1)  # decoder(encoder(y))
            test_label_y_dummy = torch.argmax(test_y_dummy, dim=1)  # decoder(y)

            test_acc_p = torch.sum(test_label_y_hat == test_label_y) / float(
                len(test_label_y)
            )
            test_acc_n = torch.sum(test_label_y_dummy != test_label_y) / float(
                len(test_label_y)
            )

            logging.info(f"autoencoder epoch:{epoch}")
            logging.info(
                f"autoencoder test acc p : {test_acc_p}, test acc n : {test_acc_n}"
            )

        model_name = f"autoencoder_{self.num_classes}_{self.hyper_lambda}"
        self.model.save_model(model_name)


class CAEFuseModelWrapper(BaseModule):
    def __init__(
        self,
        model: BaseModule,
        decoder: nn.Module,
    ):
        super().__init__()
        assert (
            model.automatic_optimization
        ), "models use MID must use automatic optimization."

        self.model = model
        self.decoder = decoder

        self._optimizers = self.model._optimizers
        self.metrics = self.model.metrics
        self.logs = self.model.logs

    def forward(self, hiddens):
        y_dummy = self.model(hiddens)
        y_dummy = F.softmax(y_dummy, dim=1)
        y_pred = self.decoder(y_dummy)
        return y_pred

    def forward_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        y_dummy = self.model(x)
        y_dummy = F.softmax(y_dummy, dim=1)
        y_pred = self.decoder(y_dummy)
        self.update_metrics(y_pred, y)

        if self.loss:
            loss = self.loss(y_pred, y)
            return y_pred, loss
        else:
            return y_pred, None


class CAEDefense(Callback):
    """Implementation of confusional autoencoder defense method in paper Defending Batch-Level Label Inference and Replacement Attacks in Vertical Federated Learning: https://ieeexplore.ieee.org/document/9833321.

    Args:
        defense_party: defense party.
        num_classes: number of label classes.
        exec_device: exec device, cpu or cuda.
        autoencoder_epochs: autoencoder training epochs.
        autoencoder_batch_size: autoencoder training batch_size.
        T: temperature before softmax to generate training/test data for autoencoder.
        hyper_lambda: entropy loss weight(lambda2 in formula 15).
        train_sample_size: training data size for autoencoder training.
        test_sample_size: test data size for autoencoder.
        learning_rate: learning rate for autoencoder training.
    """

    def __init__(
        self,
        defense_party: PYU,
        num_classes: int = 10,
        exec_device: torch.device | str = 'cpu',
        autoencoder_epochs: int = 20,
        autoencoder_batch_size: int = 128,
        T: float = 0.025,
        hyper_lambda: float = 2.0,
        train_sample_size: int = 30000,
        test_sample_size: int = 10000,
        learning_rate: float = 5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.defense_party = defense_party
        self.num_classes = num_classes
        self.exec_device = exec_device

        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.T = T
        self.hyper_lambda = hyper_lambda
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.learning_rate = learning_rate

    def on_train_begin(self, logs=None):
        def train_autoencoder(
            worker,
            num_classes,
            exec_device,
            epochs,
            batch_size,
            T,
            hyper_lambda,
            train_sample_size,
            test_sample_size,
            learning_rate,
        ):
            model = AutoEncoder(
                input_dim=num_classes, encode_dim=2 + num_classes * 6
            ).to(exec_device)

            trainer = AutoEncoderTrainer(
                model=model,
                exec_device=exec_device,
                num_classes=num_classes,
                epochs=epochs,
                batch_size=batch_size,
                T=T,
                hyper_lambda=hyper_lambda,
                train_sample_size=train_sample_size,
                test_sample_size=test_sample_size,
                learning_rate=learning_rate,
            )
            trainer.train()
            worker.autoencoder = model

        self._workers[self.defense_party].apply(
            train_autoencoder,
            self.num_classes,
            self.exec_device,
            self.autoencoder_epochs,
            self.autoencoder_batch_size,
            self.T,
            self.hyper_lambda,
            self.train_sample_size,
            self.test_sample_size,
            self.learning_rate,
        )

    def on_fuse_forward_begin(self):
        def confuse_label(worker):
            if worker.model_fuse.training:
                y = torch.unsqueeze(worker.train_y, 1)
                onehot_target = torch.zeros(y.size(0), worker.autoencoder.d).to(
                    y.device
                )
                y = onehot_target.scatter_(1, y, 1)

                _, worker.train_y = worker.autoencoder(y)

        self._workers[self.defense_party].apply(confuse_label)

    @staticmethod
    def add_decoder(worker, exec_device: torch.device):
        worker.model_fuse = CAEFuseModelWrapper(
            model=worker.model_fuse,
            decoder=worker.autoencoder.decoder,
        ).to(exec_device)

    @staticmethod
    def remove_decoder(worker):
        worker.model_fuse = worker.model_fuse.model

    def on_test_begin(self, logs=None):
        self._workers[self.defense_party].apply(self.add_decoder, self.exec_device)

    def on_test_end(self, logs=None):
        self._workers[self.defense_party].apply(self.remove_decoder)

    def on_predict_begin(self, logs=None):
        self._workers[self.defense_party].apply(self.add_decoder, self.exec_device)

    def on_predict_end(self, logs=None):
        self._workers[self.defense_party].apply(self.remove_decoder)
