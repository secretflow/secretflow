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
import math
import time
from enum import Enum, unique
from typing import List, Tuple, Union

import jax.lax
import jax.numpy as jnp
import numpy as np

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import (
    PYU,
    SPU,
    PYUObject,
    SPUCompilerNumReturnsPolicy,
    SPUObject,
    wait,
)
from secretflow.device.driver import reveal
from secretflow.ml.linear.linear_model import LinearModel, RegType
from secretflow.utils.sigmoid import SigType, sigmoid


@unique
class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'  # not supported
    L2 = 'l2'


class Strategy(Enum):
    NAIVE_SGD = 'naive_sgd'
    POLICY_SGD = 'policy_sgd'


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
    assert (
        w.shape[0] == num_feat + 1
    ), f"w shape is mismatch to x, w.shape[0]={w.shape[0]} and num_feat={num_feat}"
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


def _convergence(
    old_w: np.ndarray,
    current_w: np.ndarray,
    norm_eps: float,
    eps_scale: float,
):
    max_delta = jnp.max(jnp.abs(current_w - old_w)) * eps_scale
    max_w = jnp.max(jnp.abs(current_w))

    return (max_delta / max_w) < norm_eps


def _compute_2norm(grad, eps=1e-4):
    return jax.lax.rsqrt(jnp.sum(jnp.square(grad)) + eps)


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
    strategy: Strategy,
    dk_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    update weights on dataset in one iteration.

    Args:
        x,y: input features and label.
        w: base model weights.
        learning_rate: controls how much to change the model in one epoch.
        l2_norm: L2 regularization term.
        sig_type: sigmoid approximation type.
        reg_type: Linear or Logistic regression.
        penalty: The penalty (aka regularization term) to be used.
        total_batch: how many full batch in x.
        batch_size: how many samples use in one calculation.
        strategy: learning strategy for updating weights.
        dk_arr: only useful for policy-sgd, store the recently 1/norm(g_k) in this infeed.

    Return:
        W after update and array of norm of gradient.
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

    compute_dk = False
    if dk_arr is None and strategy == Strategy.POLICY_SGD:
        compute_dk = True
        dk_arr = []

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
        grad = jnp.matmul(jnp.transpose(x_slice), err) / batch_size

        if strategy == Strategy.POLICY_SGD:
            if compute_dk:
                scale_factor = _compute_2norm(grad)
                dk_arr.append(scale_factor)
            else:
                scale_factor = dk_arr[idx]
        else:
            scale_factor = 1

        step = learning_rate * scale_factor * grad

        if penalty == Penalty.L2:
            w_with_zero_bias = jnp.resize(w, (num_feat, 1))
            w_with_zero_bias = jnp.concatenate(
                (w_with_zero_bias, jnp.zeros((1, 1))),
                axis=0,
            )
            step = step + w_with_zero_bias * l2_norm * learning_rate / batch_size

        w = w - step

    if compute_dk:
        dk_arr = jnp.array(dk_arr)

    return w, dk_arr


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
        eps: float,
        decay_epoch: int,
        decay_rate: float,
        strategy: str,
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
        ], f"penalty should in {[e.value for e in Penalty]}, but got {penalty}"
        assert strategy in [
            e.value for e in Strategy
        ], f"strategy should in {[e.value for e in Strategy]}, but got {strategy}"

        if strategy == Strategy.POLICY_SGD:
            assert (
                reg_type == RegType.Logistic
            ), f"policy_sgd only works fine in logistic regression"

        assert eps >= 0
        if eps > 0:
            self.eps_scale = 2 ** math.floor(-math.log2(eps))
            self.norm_eps = eps * self.eps_scale

        if decay_rate is not None:
            assert 0 < decay_rate < 1, f"decay_rate should in (0, 1), got {decay_rate}"
            assert (
                decay_epoch is not None and decay_epoch > 0
            ), f"decay_epoch should > 0 if decay_rate set, got {decay_epoch}"
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch

        if strategy == Strategy.POLICY_SGD and decay_rate is None:
            # default early stop strategy for policy-sgd
            self.decay_rate = 0.5
            self.decay_epoch = 5

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
        self.strategy = Strategy(strategy)

    def _next_infeed_batch(self, ds: PYUObject, infeed_step: int) -> PYUObject:
        being = infeed_step * self.infeed_batch_size
        assert being < self.samples
        end = min(being + self.infeed_batch_size, self.samples)
        rows = end - being
        lr_total_batch = math.floor(rows / self.lr_batch_size)
        return ds[being:end], lr_total_batch

    def _get_sgd_learning_rate(self, epoch_idx: int):
        if self.decay_rate is not None:
            rate = self.decay_rate ** math.floor(epoch_idx / self.decay_epoch)
            sgd_lr = self.learning_rate * rate
        else:
            sgd_lr = self.learning_rate

        return sgd_lr

    def _epoch(self, spu_w: SPUObject, epoch_idx: int) -> SPUObject:
        """
        Complete one iteration

        Args:
            spu_w: base W to do iteration.
            epoch_idx: current epoch index.

        Return:
            W after update in SPUObject.
        """
        learning_rate = self._get_sgd_learning_rate(epoch_idx)

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

            spu_w, dk_arr = self.spu(
                _batch_update_w,
                static_argnames=(
                    'reg_type',
                    'penalty',
                    'sig_type',
                    'total_batch',
                    'batch_size',
                    'strategy',
                ),
                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                user_specified_num_returns=2,
            )(
                spu_x,
                spu_y,
                spu_w,
                learning_rate,
                self.l2_norm,
                sig_type=self.sig_type,
                reg_type=self.reg_type,
                penalty=self.penalty,
                total_batch=lr_total_batch,
                batch_size=self.lr_batch_size,
                strategy=self.strategy,
                dk_arr=self.dk_norm_dict.get(infeed_step, None),
            )
            self.dk_norm_dict[infeed_step] = dk_arr

        return spu_w

    def _convergence(self, old_w: SPUObject, current_w: SPUObject):
        spu_converged = self.spu(
            _convergence, static_argnames=('norm_eps', 'eps_scale')
        )(old_w, current_w, norm_eps=self.norm_eps, eps_scale=self.eps_scale)
        return reveal(spu_converged)

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
        eps: float = 1e-3,
        decay_epoch: int = None,
        decay_rate: float = None,
        strategy: str = 'naive_sgd',
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
            eps : float, default=1e-3
                If the W's change rate is less than this threshold, the model is considered to be converged, and the training stops early. 0 disable.
            decay_epoch / decay_rate : int, default=None
                decay learning rate, learning_rate * (decay_rate ** floor(epoch / decay_epoch)). None disable
                If strategy=policy_sgd, then decay_rate and decay_epoch have default value 0.5, 5.
            strategy : str, default=naive_sgd
                optimization strategy used in training
                  naive_sgd means origin sgd
                  policy_sgd(LR only) will scale the learning_rate in each update like adam but with unify factor,
                so the batch_size can be larger and the early stop strategy can be more aggressive, which accelerates
                training in most scenery(But not recommend for training with large regularization).
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
            eps,
            decay_epoch,
            decay_rate,
            strategy,
        )

        spu_w = self.spu(_init_w, static_argnames=('base', 'num_feat'))(
            base=0, num_feat=self.num_feat
        )

        self.batch_cache = {}
        self.dk_norm_dict = {}
        for epoch_idx in range(epochs):
            start = time.time()
            old_w = spu_w
            spu_w = self._epoch(spu_w, epoch_idx)
            wait([spu_w])
            logging.info(f"epoch {epoch_idx + 1} times: {time.time() - start}s")
            if eps > 0 and epoch_idx > 0 and self._convergence(old_w, spu_w):
                logging.info(f"early stop in {epoch_idx} epoch.")
                break

        self.batch_cache = {}
        self.dk_norm_dict = {}
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
        assert isinstance(m.weights, SPUObject), 'weights should be saved as SPUObject'
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

            to_pyu: the prediction initiator
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
