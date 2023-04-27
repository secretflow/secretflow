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
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, SPU, PYUObject, SPUObject, wait
from secretflow.device.device.type_traits import spu_fxp_precision
from secretflow.ml.linear.linear_model import LinearModel, RegType
from secretflow.utils.sigmoid import SigType, sigmoid


# hess-lr
class HESSLogisticRegression:
    """This method provides logistic regression linear models for vertical split dataset
    setting by using secret sharing and homomorphic encryption with mini batch SGD
    training solver. HESS-SGD is short for HE & secret sharing SGD training.

    During the calculation process, the HEU is used to protect the weights and
    calculate the predicted y, and the SPU is used to calculate the sigmoid and gradient.

    SPU is a verifiable and measurable secure computing device that running
    under various MPC protocols to provide provable security. More detail:
    https://www.secretflow.org.cn/docs/spu/en/

    HEU is a secure computing device that implementing HE encryption and decryption,
    and provides matrix operations similar to the numpy, reducing the threshold for use.
    More detail: https://www.secretflow.org.cn/docs/heu/en/

    For more detail, please refer to paper in KDD'21:
    https://dl.acm.org/doi/10.1145/3447548.3467210

    Args:

        spu : SPU
            SPU device.
        heu_x : HEU
            HEU device without label.
        heu_y : HEU
            HEU device with label.

    Notes:
        training dataset should be normalized or standardized,
        otherwise the SGD solver will not converge.

    """

    def __init__(self, spu: SPU, heu_x: HEU, heu_y: HEU) -> None:
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
            # HEU should work in DT_I32 mode
            heu_x.cleartext_type == "DT_I32"
            and heu_y.cleartext_type == "DT_I32"
        ), f"Heu encoding config must set to DT_I32"

        self._spu = spu
        self._heu_x = heu_x
        self._heu_y = heu_y
        self._scale = spu_fxp_precision(spu.cluster_def['runtime_config']['field'])

    def _data_check(
        self, x: Union[FedNdarray, VDataFrame]
    ) -> Tuple[PYUObject, PYUObject]:
        assert isinstance(
            x, (FedNdarray, VDataFrame)
        ), f"x should be FedNdarray or VDataFrame"
        assert len(x.partitions) == 2, f"x should be in two parties"

        x = x.values if isinstance(x, VDataFrame) else x
        assert (
            x.partition_way == PartitionWay.VERTICAL
        ), "only support vertical dataset in HESS LR"

        x_shapes = list(x.partition_shape().values())

        assert x_shapes[0][0] > x_shapes[0][1] and x_shapes[1][0] > x_shapes[1][1], (
            "samples is too small: ",
            "1. Model will not converge; 2.Y label may leak to other parties.",
        )

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

        y = y.values if isinstance(y, VDataFrame) else y

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

            x : {FedNdarray, VDataFrame}
                Input data, must be colocated with SPU.
            y : {FedNdarray, VDataFrame}
                Target data, must be located on `self._heu_y`.
            learning_rate : float, default=1e-3.
                Learning rate.
            epochs : int, default=1
                Number of epochs to train the model
            batch_size : int, default=None
                Number of samples per gradient update.
                If None, batch_size will default to number of all samples.

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
        w1 = x1.device(np.zeros)(dim1 + 1, np.int_)  # intercept
        w2 = x2.device(np.zeros)(dim2 + 1, np.int_)  # intercept
        hw1 = w1.to(self._heu_y).encrypt()
        hw2 = w2.to(self._heu_x).encrypt()

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

        self._w = self._spu(self._truncate, static_argnames=('scale'))(
            hw1.to(self._spu), hw2.to(self._spu), scale=self._scale
        )

    def save_model(self) -> LinearModel:
        """
        Save fit model in LinearModel format.
        """
        assert hasattr(self, '_w'), 'please fit model first'
        return LinearModel(self._w, RegType.Logistic, SigType.T1)

    def load_model(self, m: LinearModel) -> None:
        """
        Load LinearModel format model.
        """
        assert (
            isinstance(m.weights, SPUObject) and m.weights.device == self._spu
        ), 'weights should saved in same spu'
        self._w = m.weights
        assert m.reg_type is RegType.Logistic, 'only support Logistic Reg in HESS LR'
        assert m.sig_type is SigType.T1, 'only support T1 sigmoid in HESS LR'

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
        y = y.reshape((y.shape[0],))
        yhat = (1 << (scale - 1)) + ((p1 + p2) >> (2 * scale + 3))
        yhat = jnp.select([yhat < 0, yhat > (1 << scale)], [0, (1 << scale)], yhat)
        return yhat - y

    @staticmethod
    def _encode(x: PYUObject, scale: int, expand: bool):
        def encode(x, scale):
            if expand:
                b = np.ones((x.shape[0], 1))  # intercept
                x = np.hstack((x, b))
            return (x * scale).astype(np.int64)

        return x.device(encode)(x, scale)

    @staticmethod
    def _truncate(w1: np.ndarray, w2: np.ndarray, scale: int) -> np.ndarray:
        w1 = (w1 >> scale).astype(jnp.float32)
        w2 = (w2 >> scale).astype(jnp.float32)
        bias = jnp.full((1, 1), w1[-1] + w2[-1], dtype=jnp.float32)

        w1 = jnp.resize(w1, (w1.shape[0] - 1, 1))
        w2 = jnp.resize(w2, (w2.shape[0] - 1, 1))
        w = jnp.concatenate((w1, w2, bias)) / (1 << scale)

        return w

    @staticmethod
    def _predict(
        x: List[np.ndarray],
        w: np.ndarray,
        total_batch: int,
        batch_size: int,
    ):
        """
        predict on datasets x.

        Args:
            x: input datasets.
            w: model weights.
            total_batch: how many full batch in x.
            batch_size: how many samples use in one calculation.

        Return:
            pred scores.
        """
        x = jnp.concatenate(x, axis=1)

        num_feat = x.shape[1]
        samples = x.shape[0]
        assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w.reshape((w.shape[0], 1))

        bias = w[-1, 0]
        w = jnp.resize(w, (num_feat, 1))

        preds = []

        def get_pred(x):
            pred = jnp.matmul(x, w) + bias
            pred = sigmoid(pred, SigType.T1)
            return pred

        end = 0
        for idx in range(total_batch):
            begin = idx * batch_size
            end = (idx + 1) * batch_size
            x_slice = x[begin:end, :]
            preds.append(get_pred(x_slice))

        if end < samples:
            x_slice = x[end:samples, :]
            preds.append(get_pred(x_slice))

        return jnp.concatenate(preds, axis=0)

    def predict(self, x: Union[FedNdarray, VDataFrame]) -> PYUObject:
        """Probability estimates.

        Args:

            x : {FedNdarray, VDataFrame}
                Predict samples.

        Returns:
            PYUObject: probability of the sample for each class in the model.
        """
        ds = self._data_check(x)
        shapes = x.partition_shape()
        samples = list(shapes.values())[0][0]
        total_batch = int(samples / 1024)

        spu_yhat = self._spu(
            self._predict,
            static_argnames=('total_batch', 'batch_size'),
        )(
            [d.to(self._spu) for d in ds],
            self._w,
            total_batch=total_batch,
            batch_size=1024,
        )

        return spu_yhat
