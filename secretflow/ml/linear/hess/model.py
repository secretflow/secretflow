# Copyright 2022 Ant Group Co., Ltd.
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
from typing import Tuple, Union

import numpy as np

# from loguru import logging
from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, SPU, PYU, HEUObject, SPUObject, PYUObject, wait
from secretflow.device.device.type_traits import spu_fxp_precision


class HESSLogisticRegression:
    def __init__(self, spu: SPU, heu_x: HEU, heu_y: HEU) -> None:
        """Secure logistic regression model with homomorphic encryption and secret sharing.

        For more detail, please refer to paper in KDD'21:
        https://dl.acm.org/doi/10.1145/3447548.3467210

        Args:
            spu (SPU): SPU device.
            heu_x (HEU): HEU device without label.
            heu_y (HEU): HEU device with label.
        """
        assert (
            isinstance(spu, SPU) and len(spu.actors) == 2
        ), f"two parties support only"
        assert (
            isinstance(heu_x, HEU) and heu_x.sk_keeper_name() in spu.actors
        ), f"heu must be colocated with spu"
        assert (
            isinstance(heu_y, HEU) and heu_y.sk_keeper_name() in spu.actors
        ), f"heu must be colocated with spu"
        assert (
            heu_x.sk_keeper_name() != heu_y.sk_keeper_name()
        ), f"two heu must be provided"
        assert (
            # This type should keep same with the return type of
            # spu.Io.make_shares(). Since make_shares() always return DT_I32,
            # HEU should works in DT_I32 mode
            heu_x.cleartext_type == "DT_I32"
            and heu_y.cleartext_type == "DT_I32"
        ), f"Heu encoding config must set to DT_I32"

        self._spu = spu
        self._heu_x = heu_x
        self._heu_y = heu_y
        self._scale = spu_fxp_precision(spu.cluster_def['runtime_config']['field'])

        self._w1 = None
        self._w2 = None

    def _data_check(
        self, x: Union[FedNdarray, VDataFrame]
    ) -> Tuple[PYUObject, PYUObject]:
        assert isinstance(
            x, (FedNdarray, VDataFrame)
        ), f"x should be FedNdarray or VDataFrame"
        assert len(x.partitions) == 2, f"x should be in two parties"

        x = x.values if isinstance(x, VDataFrame) else x

        x1, x2 = None, None
        for device, partition in x.partitions.items():
            if device.party == self._heu_x.sk_keeper_name():
                x1 = partition
            elif device.party == self._heu_y.sk_keeper_name():
                x2 = partition
            else:
                raise ValueError(f"unexpected x's partition party {device.party}")

        return x1, x2

    def _args_check(self, x, y) -> Tuple[PYUObject, PYUObject, PYUObject]:
        x1, x2 = self._data_check(x)

        assert isinstance(
            y, (FedNdarray, VDataFrame)
        ), f"y should be FedNdarray or VDataFrame"
        assert len(y.partitions) == 1, f"y should be in single party"
        assert x2.device in y.partitions, f"y should be colocated with x"

        return x1, x2, y.partitions[x2.device]

    def fit(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        learning_rate=1e-3,
        epochs=1,
        batch_size=None,
    ):
        """Fit linear model with Stochastic Gradient Descent.

        Args:
            x (Union[FedNdarray, VDataFrame]): Input data, must be colocated with SPU.
            y (Union[FedNdarray, VDataFrame]): Target data, must be located on `self._heu_y`.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            epochs (int, optional): Number of epochs to train the model. Defaults to 1.
            batch_size (int, optional): Number of samples per gradient update.
            If unspecified, batch_size will default to number of all samples.
        """
        assert (
            isinstance(learning_rate, float) and learning_rate > 0
        ), f"learning_rate shoule be float > 0"
        assert isinstance(epochs, int) and epochs > 0, f"epochs should be integer > 0"
        assert batch_size is None or (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be None or integer > 0"

        x1, x2, y = self._args_check(x, y)

        x_shape = x.partition_shape()
        dim1, dim2 = x_shape[x1.device][1], x_shape[x2.device][1]
        n_samples = x_shape[x1.device][0]
        batch_size = n_samples if batch_size is None else min(n_samples, batch_size)
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        # dataset and w is scaled by driver, so devices are only aware of int type
        self._w1 = x1.device(np.zeros)(dim1 + 1, np.int_)  # intercept
        self._w2 = x2.device(np.zeros)(dim2 + 1, np.int_)  # intercept
        hw1 = self._w1.to(self._heu_y).encrypt()
        hw2 = self._w2.to(self._heu_x).encrypt()

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                logging.info(f"epoch {epoch}, step: {step} begin")

                batch_x1, batch_x2, batch_y = self._next_batch(
                    x1, x2, y, step, batch_size
                )

                # step 1. encoding features and label to fixed point integer
                scale = 1 << self._scale
                x1_ = self._encode(batch_x1, scale, True).to(self._heu_y)
                x2_ = self._encode(batch_x2, scale, True).to(self._heu_x)

                # NOTE: Gradient descent weight update:
                #       W -= learning_rate * ((h - y) @ X^T)
                # Since we do all calculations with fixed point integers, here we
                # fuse feature scale and learning_rate to avoid encoding learning_rate.
                x1_t = self._encode(batch_x1, learning_rate * scale, True).to(
                    self._heu_y
                )
                x2_t = self._encode(batch_x2, learning_rate * scale, True).to(
                    self._heu_x
                )

                y_ = self._encode(batch_y, scale, False).to(self._spu)

                # step 2. calculate loss with sigmoid minimax approximation
                p1, p2 = (x1_ @ hw1).to(self._spu), (x2_ @ hw2).to(self._spu)
                e = self._spu(self._sigmoid_minimax)(p1, p2, y_, self._scale)

                # step 3. update weights by gradient descent
                e1, e2 = e.to(self._heu_y), e.to(self._heu_x)
                hw1 -= e1 @ x1_t
                hw2 -= e2 @ x2_t

                # NOTE: Due to Ray's asynchronous scheduling, batches in different steps
                # maybe be encrypted simultaneous, which will cause memory flooding.
                wait([e])

        self._w1 = self._truncate(hw1, x1.device)
        self._w2 = self._truncate(hw2, x2.device)

    @staticmethod
    def _next_batch(x1, x2, y, step, batch_size):
        def _slice(x, step, batch_size):
            beg = step * batch_size
            end = min(beg + batch_size, len(x))
            return x[beg:end]

        batch_x1 = x1.device(_slice)(x1, step, batch_size)
        batch_x2 = x2.device(_slice)(x2, step, batch_size)
        batch_y = y.device(_slice)(y, step, batch_size)
        return batch_x1, batch_x2, batch_y

    @staticmethod
    def _sigmoid_minimax(p1, p2, y, scale):
        """sigmoid minimax approximation: y = 0.5 + 0.125 * x"""
        # NOTE: Prediction `p1` and `p2` are triple scaled integers and
        # label `y` is single scaled integer, so we have to right shift
        # `(p1 + p2)`. The whole return value is single scaled integer.
        return (1 << (scale - 1)) + ((p1 + p2) >> (2 * scale + 3)) - y

    @staticmethod
    def _encode(x: PYUObject, scale: int, expand: bool):
        def encode(x, scale):
            if expand:
                b = np.ones((x.shape[0], 1))  # intercept
                x = np.hstack((x, b))
            return (x * scale).astype(np.int64)

        return x.device(encode)(x, scale)

    def _truncate(self, w: HEUObject, device: PYU) -> SPUObject:
        # Truncate once in SPU
        w = self._spu(lambda w, scale: w >> scale)(w.to(self._spu), self._scale)
        # Truncate second time in PYU
        return device(lambda w, scale: w / (1 << scale))(w.to(device), self._scale)

    def predict(self, x: Union[FedNdarray, VDataFrame]) -> PYUObject:
        """Probability estimates.

        Args:
            x (Union[FedNdarray, VDataFrame]): Input data.

        Returns:
            PYUObject: probability of the sample for each class in the model.
        """
        x1, x2 = self._data_check(x)

        def _predict(x, w):
            b = np.ones((x.shape[0], 1))  # intercept
            x = np.hstack((x, b))
            return x @ w

        p1 = x1.device(_predict)(x1, self._w1)
        p2 = x2.device(_predict)(x2, self._w2)

        def _sigmoid(a, b):
            return 1 / (1 + np.exp(-np.add(a, b)))

        return p2.device(_sigmoid)(p1.to(p2.device), p2)
