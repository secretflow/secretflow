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

import time
import math
import logging
import numpy as np
from enum import Enum, unique
import jax.numpy as jnp
from typing import Union, List, Tuple

from secretflow.utils.sigmoid import sigmoid, SigType
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import SPU, SPUObject, wait, PYUObject, PYU
from secretflow.ml.linear.linear_model import RegType, LinearModel


@unique
class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'  # not supported
    L2 = 'l2'


'''
stateless functions use in LR training.
please keep functions stateless to make jax happy
see https://jax.readthedocs.io/en/latest/jax-101/07-state.html
'''


def _predict(
    x: List[np.ndarray],
    w: np.ndarray,
    sig_type: str,
    reg_type: str,
    total_batch: int,
    batch_size: int,
):
    """
    predict on datasets x.

    Args:
        x: input datasets.
        w: model weights.
        sig_type: sigmoid approximation type.
        reg_type: Linear or Logistic regression.
        total_batch: how many full batch in x.
        batch_size: how many samples use in one calculation.

    Return:
        pred scores.
    """
    x = jnp.concatenate(x, axis=1)

    num_feat = x.shape[1]
    samples = x.shape[0]
    assert samples >= total_batch * batch_size, "total batch is too large"
    assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
    assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
    w = w.reshape((w.shape[0], 1))

    bias = w[-1, 0]
    w = jnp.resize(w, (num_feat, 1))

    preds = []

    def get_pred(x):
        pred = jnp.matmul(x, w) + bias
        if reg_type == RegType.Logistic:
            pred = sigmoid(pred, sig_type)
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


def _concatenate(arrays: List[np.ndarray], axis: int) -> np.ndarray:
    return jnp.concatenate(arrays, axis=axis)


def _init_w(base: float, num_feat: int) -> np.ndarray:
    # last one is bias
    return jnp.full((num_feat + 1, 1), base, dtype=jnp.float32)


def _batch_update_w(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    learning_rate: float,
    l2_norm: float,
    sig_type: SigType,
    reg_type: RegType,
    penalty: Penalty,
    total_batch: int,
    batch_size: int,
) -> np.ndarray:
    """
    update weights on dataset in one iteration.

    Args:
        dataset: input datasets.
        w: base model weights.
        learning_rate: controls how much to change the model in one epoch.
        batch_size: how many samples use in one calculation.
        sig_type: sigmoid approximation type.
        reg_type: Linear or Logistic regression.
        penalty: The penalty (aka regularization term) to be used.
        l2_norm: L2 regularization term.

    Return:
        W after update.
    """
    assert x.shape[0] >= total_batch * batch_size, "total batch is too large"
    num_feat = x.shape[1]
    assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
    assert len(w.shape) == 1 or (
        len(w.shape) == 2 and w.shape[1] == 1
    ), "w should be list or 1D array"
    w = w.reshape((w.shape[0], 1))
    assert y.shape[0] == x.shape[0], "x & y not aligned"
    assert len(y.shape) == 1 or (
        len(y.shape) == 2 and y.shape[1] == 1
    ), "Y should be be list or 1D array"
    y = y.reshape((y.shape[0], 1))

    for idx in range(total_batch):
        begin = idx * batch_size
        end = (idx + 1) * batch_size
        # padding one col for bias in w
        x_slice = jnp.concatenate((x[begin:end, :], jnp.ones((batch_size, 1))), axis=1)
        y_slice = y[begin:end, :]

        pred = jnp.matmul(x_slice, w)
        if reg_type == RegType.Logistic:
            pred = sigmoid(pred, sig_type)

        err = pred - y_slice
        grad = jnp.matmul(jnp.transpose(x_slice), err)

        if penalty == Penalty.L2:
            w_with_zero_bias = jnp.resize(w, (num_feat, 1))
            w_with_zero_bias = jnp.concatenate(
                (w_with_zero_bias, jnp.zeros((1, 1))),
                axis=0,
            )
            grad = grad + w_with_zero_bias * l2_norm

        step = (learning_rate * grad) / batch_size

        w = w - step

    return w


class SSRegression:
    """
    This method provides both linear and logistic regression linear models
    for vertical split dataset setting by using secret sharing with mini
    batch SGD training solver. SS-SGD is short for secret sharing SGD training.

    more detail for SGD:
    https://stats.stackexchange.com/questions/488017/understanding-mini-batch-gradient-descent

    Linear regression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    more detail for linear regression:
    https://en.wikipedia.org/wiki/Linear_regression

    Logistic regression, despite its name, is a linear model for classification
    rather than regression. logistic regression is also known in the literature
    as logit regression, maximum-entropy classification (MaxEnt) or the log-linear
    classifier. the probabilities describing the possible outcomes of a single trial
    are modeled using a logistic function. This method can fit binary regularization
    with optional L2 regularization.

    more detail for logistic regression:
    https://en.wikipedia.org/wiki/Logistic_regression

    SPU is a verifiable and measurable secure computing device that running
    under various MPC protocols to provide provable security.

    More detail for SPU:
    https://www.secretflow.org.cn/docs/spu/en/

    This method protects the original dataset and the final model by secret sharing
    the dataset to SPU device and running model fit under SPU.

    Args:

        spu: secure device.

    Notes:
        training dataset should be normalized or standardized,
        otherwise the SGD solver will not converge.

    """

    def __init__(self, spu: SPU) -> None:
        self.spu = spu

    def _prepare_dataset(
        self, ds: Union[FedNdarray, VDataFrame]
    ) -> Tuple[FedNdarray, Tuple[int, int]]:
        """
        check data setting and get total shape.

        Args:
            ds: input dataset

        Return:
            First: dataset in unified type
            Second: shape concat all partition.
        """
        assert isinstance(
            ds, (FedNdarray, VDataFrame)
        ), f"ds should be FedNdarray or VDataFrame"
        ds = ds if isinstance(ds, FedNdarray) else ds.values
        shapes = ds.partition_shape()
        assert len(shapes) > 0, "input dataset is empty"
        assert ds.partition_way == PartitionWay.VERTICAL

        return ds, ds.shape

    def _pre_check(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        epochs: int,
        learning_rate: float,
        batch_size: int,
        sig_type: str,
        reg_type: str,
        penalty: str,
        l2_norm: float,
    ):
        """
        Parameter validity check

        Args:
            see fit()
        """
        self.x, shape = self._prepare_dataset(x)
        self.samples, self.num_feat = shape
        assert self.samples > 0 and self.num_feat > 0, "input dataset is empty"
        assert self.samples > self.num_feat, (
            "samples is too small: ",
            "1. Model will not converge; 2.Y label may leak to other parties.",
        )

        self.y, shape = self._prepare_dataset(y)
        assert self.samples == shape[0] and (
            len(shape) == 1 or shape[1] == 1
        ), "y should be list or 1D array"
        assert len(self.y.partitions) == 1

        assert epochs > 0, f"epochs should >0"
        assert learning_rate > 0, f"learning_rate should >0"
        assert batch_size > 0, f"batch_size should >0"
        assert penalty != 'l1', "not support L1 penalty for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm should >0 if use L2 penalty"

        assert sig_type in [
            e.value for e in SigType
        ], f"sig_type should in {[e.value for e in SigType]}, but got {sig_type}"
        assert reg_type in [
            e.value for e in RegType
        ], f"reg_type should in {[e.value for e in RegType]}, but got {reg_type}"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {reg_type}"

        self.lr_batch_size = batch_size
        # for large dataset, batch infeed data for each 20w*200d size.
        infeed_rows = math.ceil((200000 * 200) / self.num_feat)
        # align to lr_batch_size, for algorithm accuracy
        infeed_rows = int((infeed_rows + batch_size - 1) / batch_size) * batch_size
        self.infeed_batch_size = infeed_rows
        self.infeed_total_batch = math.ceil(self.samples / infeed_rows)
        self.learning_rate = learning_rate
        self.l2_norm = l2_norm
        self.penalty = Penalty(penalty)
        self.reg_type = RegType(reg_type)
        self.sig_type = SigType(sig_type)

    def _next_infeed_batch(self, ds: PYUObject, infeed_step: int) -> PYUObject:
        being = infeed_step * self.infeed_batch_size
        assert being < self.samples
        end = min(being + self.infeed_batch_size, self.samples)
        rows = end - being
        lr_total_batch = math.floor(rows / self.lr_batch_size)
        return ds[being:end], lr_total_batch

    def _epoch(self, spu_w: SPUObject, epoch_idx: int) -> SPUObject:
        """
        Complete one iteration

        Args:
            spu_dataset: infeed dataset.
            spu_w: base W to do iteration.
            sig_type: sigmoid approximation type.

        Return:
            W after update in SPUObject.
        """
        for infeed_step in range(self.infeed_total_batch):
            if epoch_idx == 0:
                x, lr_total_batch = self._next_infeed_batch(self.x, infeed_step)
                y, lr_total_batch = self._next_infeed_batch(self.y, infeed_step)
                spu_x = self.spu(_concatenate, static_argnames=('axis'))(
                    [x.partitions[pyu].to(self.spu) for pyu in x.partitions], axis=1
                )
                spu_y = [y.partitions[pyu].to(self.spu) for pyu in y.partitions][0]
                self.batch_cache[infeed_step] = (spu_x, spu_y, lr_total_batch)
            else:
                spu_x, spu_y, lr_total_batch = self.batch_cache[infeed_step]

            spu_w = self.spu(
                _batch_update_w,
                static_argnames=(
                    'reg_type',
                    'penalty',
                    'sig_type',
                    'total_batch',
                    'batch_size',
                ),
            )(
                spu_x,
                spu_y,
                spu_w,
                self.learning_rate,
                self.l2_norm,
                sig_type=self.sig_type,
                reg_type=self.reg_type,
                penalty=self.penalty,
                total_batch=lr_total_batch,
                batch_size=self.lr_batch_size,
            )

        return spu_w

    def fit(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        sig_type: str = 't1',
        reg_type: str = 'logistic',
        penalty: str = 'None',
        l2_norm: float = 0.5,
    ) -> None:
        """
        Fit the model according to the given training data.

        Args:

            x : {FedNdarray, VDataFrame} of shape (n_samples, n_features)
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            y : {FedNdarray, VDataFrame} of shape (n_samples,)
                Target vector relative to X.
            epochs : int
                iteration rounds.
            learning_rate : float, default=0.1
                controls how much to change the model in one epoch.
            batch_size : int, default=1024
                how many samples use in one calculation.
            sig_type : str, default=t1
                sigmoid approximation type.
            reg_type : str, default=logistic
                Linear or Logistic regression.
            penalty : str, default=None
                The penalty (aka regularization term) to be used.
            l2_norm : float, default=0.5
                L2 regularization term.

        Return:
            Final weights in SPUObject.
        """
        self._pre_check(
            x,
            y,
            epochs,
            learning_rate,
            batch_size,
            sig_type,
            reg_type,
            penalty,
            l2_norm,
        )

        spu_w = self.spu(_init_w, static_argnames=('base', 'num_feat'))(
            base=0, num_feat=self.num_feat
        )

        self.batch_cache = {}
        for epoch_idx in range(epochs):
            start = time.time()
            spu_w = self._epoch(spu_w, epoch_idx)
            wait([spu_w])
            logging.info(f"epoch {epoch_idx + 1} times: {time.time() - start}s")
            # todo: do early stop
        self.batch_cache = {}
        self.spu_w = spu_w

    def save_model(self) -> LinearModel:
        """
        Save fit model in LinearModel format.
        """
        assert hasattr(self, 'spu_w'), 'please fit model first'
        return LinearModel(self.spu_w, self.reg_type, self.sig_type)

    def load_model(self, m: LinearModel) -> None:
        """
        Load LinearModel format model.
        """
        assert (
            isinstance(m.weights, SPUObject) and m.weights.device == self.spu
        ), 'weights should saved in same spu'
        self.spu_w = m.weights
        self.reg_type = m.reg_type
        self.sig_type = m.sig_type

    def predict(
        self,
        x: Union[FedNdarray, VDataFrame],
        batch_size: int = 1024,
        to_pyu: PYU = None,
    ) -> Union[SPUObject, FedNdarray]:
        """
        Predict using the model.

        Args:

            x : {FedNdarray, VDataFrame} of shape (n_samples, n_features)
                Predict samples.

            batch_size : int, default=1024
                how many samples use in one calculation.

            to: the prediction initiator
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                otherwise, keep predict result in secret and save as SPUObject.

        Return:
            pred scores in SPUObject or FedNdarray, shape (n_samples,)
        """
        assert hasattr(self, 'spu_w'), 'please fit model first'

        x, shape = self._prepare_dataset(x)
        self.samples, self.num_feat = shape
        infeed_rows = math.ceil((200000 * 200) / self.num_feat)
        infeed_rows = int((infeed_rows + batch_size - 1) / batch_size) * batch_size
        self.infeed_batch_size = infeed_rows
        self.lr_batch_size = batch_size
        infeed_total_batch = math.ceil(self.samples / infeed_rows)

        spu_preds = []
        for infeed_step in range(infeed_total_batch):
            batch_x, lr_total_batch = self._next_infeed_batch(x, infeed_step)
            spu_x = [batch_x.partitions[pyu].to(self.spu) for pyu in batch_x.partitions]
            spu_pred = self.spu(
                _predict,
                static_argnames=('reg_type', 'sig_type', 'total_batch', 'batch_size'),
            )(
                spu_x,
                self.spu_w,
                sig_type=self.sig_type,
                reg_type=self.reg_type,
                total_batch=lr_total_batch,
                batch_size=batch_size,
            )
            spu_preds.append(spu_pred)

        pred = self.spu(_concatenate, static_argnames=('axis'))(spu_preds, axis=0)

        if to_pyu is not None:
            assert isinstance(to_pyu, PYU)
            return FedNdarray(
                partitions={
                    to_pyu: pred.to(to_pyu),
                },
                partition_way=PartitionWay.VERTICAL,
            )
        else:
            return pred
