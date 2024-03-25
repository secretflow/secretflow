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
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, SPU, PYUObject, SPUObject, wait
from secretflow.device.driver import reveal

from .core import Distribution, Linker, get_dist, get_link
from .core.distribution import DistributionBernoulli
from .metrics import BETTER_DEF, IMPROVE_DEF, SUPPORTED_METRICS, deviance

STOPPING_METRICS = list(SUPPORTED_METRICS.keys()) + ["weight"]


def _predict_on_padded_array(
    x: np.ndarray,
    o: np.ndarray,
    w: np.ndarray,
    y_scale: float,
    link: Linker,
):
    o = 0 if o is None else o
    pred = jnp.matmul(x, w) + o
    return link.response(pred) * y_scale


def _predict(
    x: List[np.ndarray],
    o: np.ndarray,
    w: np.ndarray,
    y_scale: float,
    link: Linker,
):
    x = jnp.concatenate(x, axis=1)
    num_feat = x.shape[1]
    samples = x.shape[0]
    batch_size = 1024
    total_batch = int(math.floor(samples / batch_size))
    assert (
        w.shape[0] == num_feat + 1
    ), f"w shape is mismatch to x, w {w.shape}, x {x.shape}"
    assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
    w = w.reshape((-1, 1))
    if o is not None:
        o = o.reshape((-1, 1))

    bias = w[-1, 0]
    w = jnp.resize(w, (num_feat, 1))

    preds = []

    def get_pred(x, o):
        pred = jnp.matmul(x, w) + bias + o
        return link.response(pred) * y_scale

    end = 0
    for idx in range(total_batch):
        begin = idx * batch_size
        end = begin + batch_size
        x_slice = x[begin:end, :]
        if o is not None:
            o_slice = o[begin:end, :]
            preds.append(get_pred(x_slice, o_slice))
        else:
            preds.append(get_pred(x_slice, 0))

    if end < samples:
        x_slice = x[end:samples, :]
        if o is not None:
            o_slice = o[end:samples, :]
            preds.append(get_pred(x_slice, o_slice))
        else:
            preds.append(get_pred(x_slice, 0))

    return jnp.concatenate(preds, axis=0)


def _concatenate(
    arrays: List[np.ndarray], axis: int, pad_ones: bool = False
) -> np.ndarray:
    if pad_ones:
        if axis == 1:
            ones = jnp.ones((arrays[0].shape[0], 1), dtype=arrays[0].dtype)
        else:
            ones = jnp.ones((1, arrays[0].shape[1]), dtype=arrays[0].dtype)
        arrays.append(ones)
    x = jnp.concatenate(arrays, axis=axis)
    return x


def _change_rate(old_w: np.ndarray, current_w: np.ndarray, eps_scale: float):
    if old_w is None:
        old_w = jnp.zeros(current_w.shape)

    max_delta = jnp.max(jnp.abs(current_w - old_w)) * eps_scale
    max_w = jnp.max(jnp.abs(current_w))
    return (max_delta / max_w), max_w


def _convergence(
    change_rate: np.ndarray,
    max_w: np.ndarray,
    norm_eps: float,
):
    return jnp.logical_and(change_rate < norm_eps, max_w > 0)


def _sgd_update_w(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    weight: np.ndarray,
    model: np.ndarray,
    learning_rate: float,
    link: Linker,
    dist: Distribution,
    batch_size: int,
    l2_lambda: float,
) -> np.ndarray:
    samples = x.shape[0]
    num_feat = x.shape[1]
    total_batch = int(math.floor(samples / batch_size))

    y = y.reshape((-1, 1))

    if model is None:
        model = jnp.full((num_feat, 1), 0, dtype=jnp.float64)
    else:
        model = model.reshape((-1, 1))

    if offset is not None:
        offset = offset.reshape((-1, 1))

    if weight is not None:
        weight = weight.reshape((-1, 1))

    for idx in range(total_batch):
        begin = idx * batch_size
        end = begin + batch_size
        x_slice = x[begin:end, :]
        y_slice = y[begin:end, :]

        pred = jnp.matmul(x_slice, model)
        if offset is not None:
            o_slice = offset[begin:end, :]
            pred = link.response(pred + o_slice)
        else:
            pred = link.response(pred)

        err = pred - y_slice

        if isinstance(dist, DistributionBernoulli):
            if weight is not None:
                wgt_slice = weight[begin:end, :]
                err = wgt_slice * err
            grad = jnp.matmul(jnp.transpose(x_slice), err)
            if l2_lambda is not None:
                model_l2 = model.at[-1, 0].set(0.0)
                grad = grad + l2_lambda * model_l2
            step = (learning_rate * grad) / batch_size
            model = model - step
        else:
            grad = link.response_derivative(pred)
            dev = dist.variance(pred)
            temp = grad * err / dev
            if weight is not None:
                wgt_slice = weight[begin:end, :]
                temp = wgt_slice * temp
            devp = jnp.matmul(jnp.transpose(x_slice), temp)
            if l2_lambda is not None:
                model_l2 = model.at[-1, 0].set(0.0)
                devp = devp + l2_lambda * model_l2
            model = model - learning_rate * devp / batch_size

    return model


def _irls_calculate_partials(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    weight: np.ndarray,
    model: np.ndarray,
    start_mu: np.ndarray,
    link: Linker,
    dist: Distribution,
) -> Tuple[np.ndarray, np.ndarray]:
    y = y.reshape((-1, 1))

    if offset is not None:
        offset = offset.reshape((-1, 1))

    if weight is not None:
        weight = weight.reshape((-1, 1))

    if model is not None:
        model = model.reshape((-1, 1))
        eta = jnp.matmul(x, model)
        if offset is not None:
            mu = link.response(eta + offset)
        else:
            mu = link.response(eta)
    else:
        # for correctness, start_mu should be provided
        mu = start_mu
        eta = link.link(mu)
        if offset is not None:
            eta = eta - offset

    v = dist.variance(mu)
    g_gradient = link.link_derivative(mu)
    if weight is not None:
        W_diag = weight / dist.scale() / (v * g_gradient) / g_gradient
    else:
        W_diag = 1 / dist.scale() / (v * g_gradient) / g_gradient
    Z = eta + (y - mu) * g_gradient

    XTW = jnp.transpose(x * W_diag.reshape(-1, 1))
    J = jnp.matmul(XTW, x)
    XTWZ = jnp.matmul(XTW, Z)
    return J, XTWZ


def J_inv(J, l2_lambda):
    if l2_lambda is not None:
        I_m = np.identity(J.shape[0])
        I_m[-1, -1] = 0.0
        J = J + l2_lambda * I_m

    inv_J = np.linalg.inv(J)
    return inv_J


def _irls_update_w_from_accumulated_partials(inv_J, XTWZ, model, l2_lambda):
    if l2_lambda and model is not None:
        model_l2 = model.at[-1, 0].set(0.0)
        model = jnp.matmul(inv_J, XTWZ - l2_lambda * model_l2)
    else:
        model = jnp.matmul(inv_J, XTWZ)

    return model


class SSGLM:
    def __init__(self, spu: SPU) -> None:
        self.spu = spu

    def _prepare_dataset(
        self, ds: Union[FedNdarray, VDataFrame]
    ) -> Tuple[FedNdarray, Tuple[int, int]]:
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
        o: Union[FedNdarray, VDataFrame],
        w: Union[FedNdarray, VDataFrame],
        irls_epochs: int,
        sgd_epochs: int,
        link: str,
        dist: str,
        power: float,
        scale: float,
        sgd_learning_rate: float,
        sgd_batch_size: int,
        decay_epoch: int,
        decay_rate: float,
        l2_lambda: float,
        infeed_batch_size_limit: int,
        fraction_of_validation_set: float,
        stopping_metric: str,
        stopping_rounds: int,
        stopping_tolerance: float,
        random_state: int,
    ):
        # early stop settings check
        if stopping_rounds > 0:
            assert (
                0 < fraction_of_validation_set < 1
            ), f"validation fraction must be in (0, 1), got {fraction_of_validation_set}"
            assert (
                stopping_metric in STOPPING_METRICS
            ), f"invalid metric {stopping_metric}, must be one of {STOPPING_METRICS}"
            if stopping_metric == "weight":
                fraction_of_validation_set = 0
            assert (
                stopping_tolerance > 0
            ), f"tolerance must be positive, got {stopping_tolerance}"

            if stopping_metric == "AUC":
                assert link == "logit", "only logit link supports AUC metric"

            if fraction_of_validation_set > 0:
                x, x_val = train_test_split(
                    x, test_size=fraction_of_validation_set, random_state=random_state
                )
                y, y_val = train_test_split(
                    y, test_size=fraction_of_validation_set, random_state=random_state
                )

                if w is not None:
                    w, w_val = train_test_split(
                        w,
                        test_size=fraction_of_validation_set,
                        random_state=random_state,
                    )

                if o is not None:
                    o, o_val = train_test_split(
                        o,
                        test_size=fraction_of_validation_set,
                        random_state=random_state,
                    )
        self.x, (self.samples, self.num_feat) = self._prepare_dataset(x)
        assert self.samples > 0 and self.num_feat > 0, "input dataset is empty"
        assert self.samples > self.num_feat, (
            "samples is too small: ",
            "1. Model will not converge; 2.Y label may leak to other parties.",
        )
        assert (
            sgd_epochs == 0 or self.samples >= sgd_batch_size
        ), f"batch_size {sgd_batch_size} is too large for training dataset samples {self.samples}"
        validation_set_not_empty = (
            stopping_rounds > 0 and fraction_of_validation_set > 0
        )
        if validation_set_not_empty:
            self.x_val, (self.samples_val, _) = self._prepare_dataset(x_val)

        self.y, shape = self._prepare_dataset(y)
        if validation_set_not_empty:
            self.y_val, _ = self._prepare_dataset(y_val)

        assert self.samples == shape[0] and (
            len(shape) == 1 or shape[1] == 1
        ), "y should be list or 1D array"
        assert len(self.y.partitions) == 1

        def normalize_y(y: np.ndarray) -> Tuple[np.ndarray, float]:
            y_scale = y.max() / 2
            if y_scale > 1:
                return y / y_scale, y_scale
            else:
                return y, 1

        y_device = list(self.y.partitions.keys())[0]
        self.y_device = y_device

        y_object_not_scaled = self.y.partitions[y_device]
        y, y_scale = y_device(normalize_y)(y_object_not_scaled)
        self.y_object_not_scaled = y_object_not_scaled

        self.y.partitions[y_device] = y
        if validation_set_not_empty:
            y_val_object_not_scaled = self.y_val.partitions[y_device]
            self.y_val.partitions[y_device] = y_device(lambda y, scale: y / scale)(
                y_val_object_not_scaled, y_scale
            )
            self.y_val_object_not_scaled = y_val_object_not_scaled
        self.y_scale = reveal(y_scale)
        self.y_device = y_device

        if o is not None:
            self.offset, shape = self._prepare_dataset(o)
            assert self.samples == shape[0] and (
                len(shape) == 1 or shape[1] == 1
            ), "offset should be list or 1D array"
            assert len(self.offset.partitions) == 1
            if validation_set_not_empty:
                self.offset_val, _ = self._prepare_dataset(o_val)
        else:
            self.offset = None
            self.offset_val = None

        if w is not None:
            self.weight, shape = self._prepare_dataset(w)
            assert self.samples == shape[0] and (
                len(shape) == 1 or shape[1] == 1
            ), "weight should be list or 1D array"
            assert len(self.weight.partitions) == 1
            if validation_set_not_empty:
                self.weight_val, _ = self._prepare_dataset(w_val)
        else:
            self.weight = None
            self.weight_val = None

        assert (
            irls_epochs >= 0 and sgd_epochs >= 0 and (irls_epochs + sgd_epochs) > 0
        ), f"epochs should >0"
        self.irls_epochs = irls_epochs
        self.sgd_epochs = sgd_epochs
        self.epochs = irls_epochs + sgd_epochs
        assert sgd_learning_rate > 0, f"learning_rate should >0"
        self.sgd_learning_rate = sgd_learning_rate

        self.link = get_link(link)
        self.dist = get_dist(dist, scale, power)

        assert stopping_tolerance >= 0
        if stopping_tolerance > 0:
            self.eps_scale = 2 ** math.floor(-math.log2(stopping_tolerance))
            self.norm_eps = stopping_tolerance * self.eps_scale

        assert sgd_batch_size > 0, f"sgd_batch_size should >0"
        self.sgd_batch_size = sgd_batch_size
        # for large dataset, batch infeed data for each 10w*100d size.
        infeed_rows = math.ceil(infeed_batch_size_limit / self.num_feat)
        # align to sgd_batch_size, for algorithm accuracy
        infeed_rows = (
            int((infeed_rows + sgd_batch_size - 1) / sgd_batch_size) * sgd_batch_size
        )
        self.infeed_batch_size = infeed_rows
        self.infeed_total_batch = math.ceil(self.samples / infeed_rows)
        if validation_set_not_empty:
            self.infeed_total_batch_val = math.ceil(self.samples_val / infeed_rows)

        if decay_rate is not None:
            assert (
                0 < decay_rate and decay_rate < 1
            ), f"decay_rate should in (0, 1), got {decay_rate}"
            assert (
                decay_epoch is not None and decay_epoch > 0
            ), f"decay_epoch should > 0 if decay_rate set, got {decay_epoch}"
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch

        if l2_lambda is not None:
            assert 0 < l2_lambda, f"l2_lambda should be greater than 0, got {l2_lambda}"
        self.l2_lambda = l2_lambda

    def _sgd_step(
        self,
        spu_model: SPUObject,
        spu_x: SPUObject,
        spu_y: SPUObject,
        spu_o: SPUObject,
        spu_w: SPUObject,
        learning_rate: float,
    ) -> SPUObject:
        spu_model = self.spu(
            _sgd_update_w,
            static_argnames=(
                'learning_rate',
                'link',
                'dist',
                'batch_size',
                'l2_lambda',
            ),
        )(
            spu_x,
            spu_y,
            spu_o,
            spu_w,
            spu_model,
            learning_rate=learning_rate,
            link=self.link,
            dist=self.dist,
            batch_size=self.sgd_batch_size,
            l2_lambda=self.l2_lambda,
        )

        return spu_model

    def _next_infeed_batch(
        self,
        ds: Union[SPUObject, FedNdarray, PYUObject],
        infeed_step: int,
        samples: int = None,
        infeed_batch_size: int = None,
    ) -> Union[SPUObject, FedNdarray, PYUObject]:
        if samples is None:
            samples = self.samples
        if infeed_batch_size is None:
            infeed_batch_size = self.infeed_batch_size

        begin = infeed_step * infeed_batch_size
        assert begin < samples
        end = min(begin + infeed_batch_size, samples)
        if isinstance(ds, FedNdarray):
            return ds[begin:end]
        elif isinstance(ds, SPUObject):
            return self.spu(lambda ds: ds[begin:end])(ds)
        elif isinstance(ds, PYUObject):
            return ds.device(lambda x, begin, end: x[begin:end])(ds, begin, end)
        else:
            raise TypeError(f'unsupported type of ds: {type(ds)}')

    def _to_spu(self, d: FedNdarray):
        return [d.partitions[pyu].to(self.spu) for pyu in d.partitions]

    def _build_batch_cache(self, infeed_step: int, cache_name: str = "train"):
        if cache_name == "train":
            x = self.x
            y = self.y
            offset = self.offset
            weight = self.weight
            samples = self.samples
        elif cache_name == "val":
            x = self.x_val
            y = self.y_val
            offset = self.offset_val
            weight = self.weight_val
            samples = self.samples_val
        else:
            raise NotImplementedError("only train/val cache supported")

        x = self._next_infeed_batch(x, infeed_step, samples)
        y = self._next_infeed_batch(y, infeed_step, samples)

        spu_x = self.spu(_concatenate, static_argnames=('axis', 'pad_ones'))(
            self._to_spu(x), axis=1, pad_ones=True
        )
        spu_y = self._to_spu(y)[0]

        if offset is not None:
            o = self._next_infeed_batch(offset, infeed_step, samples)
            spu_o = self._to_spu(o)[0]
        else:
            spu_o = None

        if weight is not None:
            w = self._next_infeed_batch(weight, infeed_step, samples)
            spu_w = self._to_spu(w)[0]
        else:
            spu_w = None

        self.batch_cache[cache_name][infeed_step] = (spu_x, spu_y, spu_o, spu_w)

    def _get_sgd_learning_rate(self, epoch_idx: int):
        if self.decay_rate is not None:
            rate = self.decay_rate ** math.floor(epoch_idx / self.decay_epoch)
            sgd_lr = self.sgd_learning_rate * rate
        else:
            sgd_lr = self.sgd_learning_rate

        return sgd_lr

    def _epoch(self, spu_model: SPUObject, epoch_idx: int) -> SPUObject:
        dist = self.dist
        start_mu_slice = None
        if epoch_idx == 0:
            y = self.y.partitions[self.y_device]
            start_mu = self.y_device(
                lambda dist, y: dist.starting_mu(y).reshape(-1, 1)
            )(dist, y)
        if epoch_idx < self.irls_epochs:
            for infeed_step in range(self.infeed_total_batch):
                if epoch_idx == 0:
                    self._build_batch_cache(infeed_step)
                    start_mu_slice = self._next_infeed_batch(start_mu, infeed_step)
                spu_x, spu_y, spu_o, spu_w = self.batch_cache["train"][infeed_step]
                logging.info("irls calculating partials...")
                new_J, new_XTWZ = self.spu(
                    _irls_calculate_partials,
                    static_argnames=(
                        'link',
                        'dist',
                    ),
                    num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_COMPILER,
                )(
                    spu_x,
                    spu_y,
                    spu_o,
                    spu_w,
                    spu_model,
                    start_mu_slice,
                    link=self.link,
                    dist=self.dist,
                )
                wait([new_J, new_XTWZ])
                if infeed_step == 0:
                    J = new_J
                    XTWZ = new_XTWZ
                else:
                    J, XTWZ = self.spu(
                        lambda x, y, z, w: (x + y, z + w),
                        num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
                        user_specified_num_returns=2,
                    )(new_J, J, new_XTWZ, XTWZ)
                    wait([J, XTWZ])
            # it is safe to reveal J to y here
            inv_J = self.y_device(J_inv)(J.to(self.y_device), self.l2_lambda)
            logging.info("irls updating weights...")
            spu_model = self.spu(
                _irls_update_w_from_accumulated_partials,
                static_argnames=('l2_lambda',),
            )(
                inv_J.to(self.spu),
                XTWZ,
                spu_model,
                l2_lambda=self.l2_lambda,
            )
        else:
            for infeed_step in range(self.infeed_total_batch):
                if epoch_idx == 0:
                    self._build_batch_cache(infeed_step)
                spu_x, spu_y, spu_o, spu_w = self.batch_cache["train"][infeed_step]
                sgd_lr = self._get_sgd_learning_rate(epoch_idx - self.irls_epochs)
                spu_model = self._sgd_step(
                    spu_model,
                    spu_x,
                    spu_y,
                    spu_o,
                    spu_w,
                    sgd_lr,
                )

        return spu_model

    def _convergence(
        self,
        old_w: SPUObject,
        current_w: SPUObject,
        epoch_idx: int,
        report_metric: bool = False,
    ):
        change_rate, max_w = self.spu(
            _change_rate,
            static_argnames=('eps_scale'),
            num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=2,
        )(old_w, current_w, eps_scale=self.eps_scale)
        if report_metric:
            change_rate = reveal(change_rate)
            logging.info(f"epoch {epoch_idx}: change rate: {change_rate}")
            self.train_metric_history.append(
                {
                    "epoch": epoch_idx + 1,
                    "stopping_metric_type": "weight change rate",
                    "train_metric_value": change_rate,
                }
            )

        spu_converged = self.spu(_convergence, static_argnames=('norm_eps'))(
            change_rate, max_w, norm_eps=self.norm_eps
        )
        return reveal(spu_converged)

    def _metric(
        self,
        stopping_metric: str,
        dist: Distribution,
        epoch: int,
        dataset_type: str = 'val',
    ):
        assert dataset_type in ['val', 'train']
        infeed_total_batch = (
            self.infeed_total_batch_val
            if dataset_type == 'val'
            else self.infeed_total_batch
        )
        samples = self.samples_val if dataset_type == 'val' else self.samples
        assert stopping_metric in SUPPORTED_METRICS
        if epoch == 0:
            for infeed_step in range(infeed_total_batch):
                self._build_batch_cache(infeed_step, cache_name=dataset_type)
        y_pred = self._predict_on_dataset(dataset=dataset_type)

        if stopping_metric == 'deviance':
            # deviance can be calculated by y device if weight is already at the label holder device
            if (
                self.weight is None
                or list(self.weight.partitions.keys())[0] == self.y_device
            ):
                y = (
                    self.y_val_object_not_scaled
                    if dataset_type == 'val'
                    else self.y_object_not_scaled
                )
                if self.weight is None:
                    weight = None
                else:
                    weight = (
                        list(self.weight_val.partitions.values())[0]
                        if dataset_type == 'val'
                        else list(self.weight.partitions.values())[0]
                    )
                metric = self.y_device(deviance)(
                    y,
                    y_pred.to(self.y_device),
                    weight,
                    dist,
                )
            # deviance should be calculated by spu if weight is at the non-label holder device
            else:
                metric = 0
                # deviance can be calculated by batches
                for infeed_step in range(infeed_total_batch):
                    _, spu_y, _, spu_w = self.batch_cache[dataset_type][infeed_step]
                    spu_y_pred = self._next_infeed_batch(y_pred, infeed_step, samples)
                    spu_y = self.spu(lambda y, scale: y * scale)(spu_y, self.y_scale)

                    metric_ = self.spu(deviance, static_argnames=('dist'))(
                        spu_y,
                        spu_y_pred,
                        spu_w,
                        dist=dist,
                    )
                    metric = self.spu(lambda x, y: x + y)(metric, metric_)
        # in all other cases, metric can be calculated by y device
        else:
            y = (
                self.y_val_object_not_scaled
                if dataset_type == 'val'
                else self.y_object_not_scaled
            )
            metric = self.y_device(SUPPORTED_METRICS[stopping_metric])(
                y, y_pred.to(self.y_device)
            )
        # we have deviced to reveal the metric
        return reveal(metric)

    def _check_early_stop(
        self,
        stopping_metric,
        stopping_tolerance,
        old_w,
        spu_w,
        stopping_rounds,
        dist,
        scale,
        power,
        epoch_idx,
        report_metric,
    ):
        if (
            stopping_metric == "weight"
            and stopping_tolerance > 0
            and self._convergence(old_w, spu_w, epoch_idx, report_metric)
        ):
            logging.info("early stop")
            return True

        if stopping_metric != "weight" and stopping_rounds > 0:
            self.spu_w = spu_w
            start = time.time()
            dist_ = get_dist(dist, scale, power)
            metric = self._metric(
                stopping_metric,
                dist_,
                epoch_idx,
                dataset_type='val',
            )

            if report_metric:
                train_metric = self._metric(
                    stopping_metric,
                    dist_,
                    epoch_idx,
                    dataset_type='train',
                )

            if self.best_metric is not None:
                self.improvement_history[epoch_idx % stopping_rounds] = IMPROVE_DEF[
                    stopping_metric
                ](self.best_metric, metric)

            stopped = (
                False
                if epoch_idx < stopping_rounds
                else np.all(self.improvement_history <= stopping_tolerance)
            )

            if self.best_metric is None:
                self.best_metric = metric
                self.best_spu_w = spu_w
            else:
                remain_best = BETTER_DEF[stopping_metric](self.best_metric, metric)
                self.best_metric = self.best_metric if remain_best else metric
                self.best_spu_w = self.best_spu_w if remain_best else spu_w
            validation_time_cost = time.time() - start
            logging.info(
                f"epoch {epoch_idx + 1} validation time cost: {validation_time_cost}"
            )
            if report_metric:
                logging.info(
                    f'stopping_metric_type {stopping_metric}:train {train_metric:.6f}, validation {metric:.6f}'
                )
                self.train_metric_history.append(
                    {
                        "epoch": epoch_idx + 1,
                        "stopping_metric_type": stopping_metric,
                        "train_metric_value": train_metric,
                        "validation_metric_value": metric,
                        "validation_time_cost": validation_time_cost,
                    }
                )
            return stopped
        return False

    def _fit(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        offset: Union[FedNdarray, VDataFrame],
        weight: Union[FedNdarray, VDataFrame],
        link: str,
        dist: str,
        power: float,
        irls_epochs: int = 0,
        sgd_epochs: int = 0,
        scale: float = 1,
        sgd_learning_rate: float = 0.1,
        sgd_batch_size: int = 1024,
        decay_epoch: int = None,
        decay_rate: float = None,
        l2_lambda: float = None,
        # 10w * 100d
        infeed_batch_size_limit: int = 10000000,
        fraction_of_validation_set: float = 0.2,
        stopping_metric: str = 'deviance',
        stopping_rounds: int = 0,
        stopping_tolerance: float = 0.001,
        report_metric: bool = False,
        random_state: int = 1212,
    ) -> None:
        self._pre_check(
            x,
            y,
            offset,
            weight,
            irls_epochs,
            sgd_epochs,
            link,
            dist,
            power,
            scale,
            sgd_learning_rate,
            sgd_batch_size,
            decay_epoch,
            decay_rate,
            l2_lambda,
            infeed_batch_size_limit,
            fraction_of_validation_set,
            stopping_metric,
            stopping_rounds,
            stopping_tolerance,
            random_state,
        )

        spu_w = None

        self.batch_cache = {"train": {}, "val": {}}
        if report_metric:
            self.train_metric_history = []

        if stopping_rounds > 0 and stopping_metric != "weight":
            self.improvement_history = np.zeros((stopping_rounds,))
            self.best_metric = None
            self.best_spu_w = None

        for epoch_idx in range(self.epochs):
            start = time.time()
            old_w = spu_w
            spu_w = self._epoch(spu_w, epoch_idx)
            wait([spu_w])
            logging.info(f"epoch {epoch_idx + 1} train times: {time.time() - start}s")
            stopped = self._check_early_stop(
                stopping_metric,
                stopping_tolerance,
                old_w,
                spu_w,
                stopping_rounds,
                dist,
                scale,
                power,
                epoch_idx,
                report_metric,
            )
            if stopped:
                # not improving by tolerance for all stopping_rounds, stop!
                logging.info("early stop triggered")
                spu_w = self.best_spu_w
                break

        self.batch_cache = {"train": {}, "val": {}}
        self.spu_w = spu_w

    def fit_irls(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        offset: Union[FedNdarray, VDataFrame],
        weight: Union[FedNdarray, VDataFrame],
        epochs: int,
        link: str,
        dist: str,
        tweedie_power: float = 1,
        scale: float = 1,
        l2_lambda: float = None,
        # 10w * 100d
        infeed_batch_size_limit: int = 10000000,
        fraction_of_validation_set: float = 0.2,
        random_state: int = 1212,
        stopping_metric: str = 'deviance',
        stopping_rounds: int = 0,
        stopping_tolerance: float = 0.001,
        report_metric: bool = False,
    ) -> None:
        """
        Fit the model by IRLS(Iteratively reweighted least squares).

        Args:

            x : {FedNdarray, VDataFrame} of shape (n_samples, n_features)
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            y : {FedNdarray, VDataFrame} of shape (n_samples,)
                Target vector relative to X.
            offset : {FedNdarray, VDataFrame} of shape (n_samples,)
                Specify a column to use as the offset, Offsets are per-row “bias values” that are used during model training.
            weight : {FedNdarray, VDataFrame} of shape (n_samples,)
                Specify a column to use for the observation weights, which are used for bias correction.
            epochs : int
                iteration rounds.
            link : str
                Specify a link function (Logit, Log, Reciprocal, Identity)
            dist : str
                Specify a probability distribution (Bernoulli, Poisson, Gamma, Tweedie)
            tweedie_power : float
                Tweedie distributions are a family of distributions that include normal, gamma, poisson and their combinations.
                    0: Specialized as normal
                    1: Specialized as poisson
                    2: Specialized as gamma
                    (1,2): combinations of gamma and poisson
            scale : float
                A guess value for distribution's scale.
            learning_rate : float, default=0.1
                controls how much to change the model in one epoch.
            batch_size : int, default=1024
                how many samples use in one calculation.
            iter_start_irls : int, default=0
                run a few rounds of irls training as the initialization of w, 0 disable.
            l2_lambda: float, default=None
                the coefficient of L2 regularization loss is 1/2 * l2_lambda. It needs to be greater than 0.
            fraction_of_validation_set : float, default=0.2.
            random_state: int, default=1212. random state for validation split.
            stopping_metric: float, default='deviance'. must be one of deviance, weight, AUC, RMSE, MSE.
            stopping_rounds: int, default=0. If the model is not improving for stopping_rounds, the training process will be stopped,
            for 'weight' stopping metric, stopping_rounds is fixed to be 1
            stopping_tolerance: float, default=0.001. the model is considered as not improving, if the metric is not improved by tolerance over best metric in history.
            report_metric: bool, default=False. Whether to report the value of stopping metric. Not effective for weight change rate.
        """
        self._fit(
            x,
            y,
            offset,
            weight,
            link,
            dist,
            tweedie_power,
            irls_epochs=epochs,
            scale=scale,
            l2_lambda=l2_lambda,
            infeed_batch_size_limit=infeed_batch_size_limit,
            fraction_of_validation_set=fraction_of_validation_set,
            random_state=random_state,
            stopping_metric=stopping_metric,
            stopping_rounds=stopping_rounds,
            stopping_tolerance=stopping_tolerance,
            report_metric=report_metric,
        )

    def fit_sgd(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        offset: Union[FedNdarray, VDataFrame],
        weight: Union[FedNdarray, VDataFrame],
        epochs: int,
        link: str,
        dist: str,
        tweedie_power: float = 1,
        scale: float = 1,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        iter_start_irls: int = 0,
        decay_epoch: int = None,
        decay_rate: float = None,
        l2_lambda: float = None,
        infeed_batch_size_limit: int = 10000000,
        fraction_of_validation_set: float = 0.2,
        random_state: int = 1212,
        stopping_metric: str = 'deviance',
        stopping_rounds: int = 0,
        stopping_tolerance: float = 0.001,
        report_metric: bool = False,
    ) -> None:
        """
        Fit the model by SGD(stochastic gradient descent).

        Args:

            x : {FedNdarray, VDataFrame} of shape (n_samples, n_features)
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            y : {FedNdarray, VDataFrame} of shape (n_samples,)
                Target vector relative to X.
            offset : {FedNdarray, VDataFrame} of shape (n_samples,)
                Specify a column to use as the offset, Offsets are per-row “bias values” that are used during model training.
            weight : {FedNdarray, VDataFrame} of shape (n_samples,)
                Specify a column to use for the observation weights, which are used for bias correction.
            epochs : int
                iteration rounds.
            link : str
                Specify a link function (Logit, Log, Reciprocal, Identity)
            dist : str
                Specify a probability distribution (Bernoulli, Poisson, Gamma, Tweedie)
            tweedie_power : float
                Tweedie distributions are a family of distributions that include normal, gamma, poisson and their combinations.
                    0: Specialized as normal
                    1: Specialized as poisson
                    2: Specialized as gamma
                    (1,2): combinations of gamma and poisson
            scale : float
                A guess value for distribution's scale.
            learning_rate : float, default=0.1
                controls how much to change the model in one epoch.
            batch_size : int, default=1024
                how many samples use in one calculation.
            iter_start_irls : int, default=0
                run a few rounds of irls training as the initialization of w, 0 disable.
            decay_epoch / decay_rate : int, default=None
                decay learning rate, learning_rate * (decay_rate ** floor(epoch / decay_epoch)). None disable
            l2_lambda: float, default=None
                the coefficient of L2 regularization loss is 1/2 * l2_lambda. It needs to be greater than 0.
            fraction_of_validation_set : float, default=0.2.
            random_state: int, default=1212. random state for validation split.
            stopping_rounds: int, default=0. The moving average is calculated over the last stopping_rounds rounds,
            for 'weight' stopping metric, stopping_rounds is fixed to be 1.
            stopping_tolerance: float, default=0.001. the model will stop if the ratio between the best moving average and reference moving average is less than 1 - tolerance
            report_metric: bool, default=False. Whether to report the value of stopping metric. Not effective for weight change rate.
        """
        self._fit(
            x,
            y,
            offset,
            weight,
            link,
            dist,
            tweedie_power,
            irls_epochs=iter_start_irls,
            sgd_epochs=epochs,
            scale=scale,
            sgd_learning_rate=learning_rate,
            sgd_batch_size=batch_size,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
            l2_lambda=l2_lambda,
            infeed_batch_size_limit=infeed_batch_size_limit,
            fraction_of_validation_set=fraction_of_validation_set,
            random_state=random_state,
            stopping_metric=stopping_metric,
            stopping_rounds=stopping_rounds,
            stopping_tolerance=stopping_tolerance,
            report_metric=report_metric,
        )

    def _predict_on_dataset(self, dataset: str = 'val') -> Union[SPUObject, PYUObject]:
        assert hasattr(self, 'spu_w'), 'please fit model first'
        assert hasattr(self, 'link'), 'please fit model first'
        assert hasattr(self, 'y_scale'), 'please fit model first'

        assert dataset in ['train', 'val']
        infeed_total_batch = (
            self.infeed_total_batch_val if dataset == 'val' else self.infeed_total_batch
        )

        spu_preds = []
        for infeed_step in range(infeed_total_batch):
            assert (
                infeed_step in self.batch_cache[dataset]
            ), f'{dataset} batch {infeed_step} not found'
            spu_x, _, spu_o, _ = self.batch_cache[dataset][infeed_step]
            spu_pred = self.spu(
                _predict_on_padded_array,
                static_argnames=('link', 'y_scale'),
            )(
                spu_x,
                spu_o,
                self.spu_w,
                y_scale=self.y_scale,
                link=self.link,
            )
            spu_preds.append(spu_pred)

        pred = self.spu(lambda p: jnp.concatenate(p, axis=0))(spu_preds)
        return pred

    def predict(
        self,
        x: Union[FedNdarray, VDataFrame],
        o: Union[FedNdarray, VDataFrame] = None,
        to_pyu: PYU = None,
        # 10w * 100d
        infeed_batch_size_limit: int = 10000000,
    ) -> Union[SPUObject, PYUObject]:
        """
        Predict using the model.

        Args:

            x : {FedNdarray, VDataFrame} of shape (n_samples, n_features)
                Predict samples.
            o : {FedNdarray, VDataFrame} of shape (n_samples,)
                Specify a column to use as the offset as per-row “bias values” use in predict
            to_pyu : the prediction initiator
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                otherwise, keep predict result in secret and save as SPUObject.

        Return:
            pred scores in SPUObject, shape (n_samples,)
        """
        assert hasattr(self, 'spu_w'), 'please fit model first'
        assert hasattr(self, 'link'), 'please fit model first'
        assert hasattr(self, 'y_scale'), 'please fit model first'

        x, shape = self._prepare_dataset(x)
        if o is not None:
            o, _ = self._prepare_dataset(o)
        samples, num_feat = shape
        infeed_rows = math.ceil(infeed_batch_size_limit / num_feat)
        infeed_batch_size = infeed_rows
        infeed_total_batch = math.ceil(samples / infeed_rows)

        spu_preds = []
        for infeed_step in range(infeed_total_batch):
            batch_x = self._next_infeed_batch(
                x, infeed_step, samples, infeed_batch_size
            )
            spu_x = self._to_spu(batch_x)
            if o is not None:
                batch_o = self._next_infeed_batch(
                    o, infeed_step, samples, infeed_batch_size
                )
                spu_o = self._to_spu(batch_o)[0]
            else:
                spu_o = None
            spu_pred = self.spu(
                _predict,
                static_argnames=('link', 'y_scale'),
            )(
                spu_x,
                spu_o,
                self.spu_w,
                y_scale=self.y_scale,
                link=self.link,
            )
            spu_preds.append(spu_pred)

        pred = self.spu(lambda p: jnp.concatenate(p, axis=0))(spu_preds)

        if to_pyu is not None:
            assert isinstance(to_pyu, PYU), f"to_pyu must be a PYU, got {type(to_pyu)}"
            return FedNdarray(
                partitions={
                    to_pyu: pred.to(to_pyu),
                },
                partition_way=PartitionWay.VERTICAL,
            )
        else:
            return pred

    def spu_w_to_federated(
        self, federated_template: Union[FedNdarray, VDataFrame], bias_receiver: PYU
    ) -> Tuple[FedNdarray, PYUObject]:
        """spu_w is our trained model of shape (num_feature + 1, 1)
        we are going to split it into federated form.

        Args:
            federated_template : FedNdarray or VDataFrame. Training data or test data X will do. Used as template to split W.
            bias_receiver: PYU. Specify which party receive the bias term.
        """
        federated_template, (_, num_feat) = self._prepare_dataset(federated_template)
        assert (
            self.num_feat == num_feat
        ), f"federated template must have number of features equal {self.num_feat}"
        cum_index = 0
        spu_w = self.spu(lambda x: x.reshape(-1, 1))(self.spu_w)
        fed_w = {}
        for party_device, shape in federated_template.partition_shape().items():
            party_feat_num = shape[1]
            beg = cum_index
            end = cum_index + party_feat_num
            fed_w[party_device] = self.spu(
                slice_x,
                static_argnames=('beg', 'end'),
            )(
                spu_w, beg=beg, end=end
            ).to(party_device)
            cum_index = end
        return FedNdarray(
            partitions=fed_w, partition_way=PartitionWay.VERTICAL
        ), self.spu(
            slice_x,
            static_argnames=('beg', 'end'),
        )(
            spu_w, beg=num_feat, end=num_feat + 1
        ).to(
            bias_receiver
        )

    def predict_fed_w(
        self,
        x: Union[FedNdarray, VDataFrame],
        fed_w: FedNdarray,
        bias: PYUObject,
        o: Union[FedNdarray, VDataFrame] = None,
        to_pyu: PYU = None,
        infeed_batch_size_limit: int = 10000000,
    ) -> PYUObject:
        """
        Predict using the model in a federated form, suppose all slices collected by to_pyu device.
        Not MPC prediction.

        Args:

            x : {FedNdarray, VDataFrame} of shape (n_samples, n_features)
                Predict samples.
            fed_w: FedNdarray. w in a federated form.
            bias: PYUObject. bias term.
            o : {FedNdarray, VDataFrame} of shape (n_samples,).
                Specify a column to use as the offset as per-row “bias values” use in predict
            to_pyu : the prediction initiator.
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                Default to be bias holder device.


        Return:
            pred scores in PYUObject, shape (n_samples,)
        """
        assert hasattr(self, 'spu_w'), 'please fit model first'
        assert hasattr(self, 'link'), 'please fit model first'
        assert hasattr(self, 'y_scale'), 'please fit model first'

        x, shape = self._prepare_dataset(x)
        if o is not None:
            o, _ = self._prepare_dataset(o)
        samples, num_feat = shape
        infeed_rows = math.ceil(infeed_batch_size_limit / num_feat)
        infeed_batch_size = infeed_rows
        infeed_total_batch = math.ceil(samples / infeed_rows)
        if to_pyu is None:
            to_pyu = bias.device

        preds = []
        for infeed_step in range(infeed_total_batch):
            batch_x = self._next_infeed_batch(
                x, infeed_step, samples, infeed_batch_size
            )
            if o is not None:
                batch_o = self._next_infeed_batch(
                    o, infeed_step, samples, infeed_batch_size
                )
            else:
                batch_o = None
            dot_products = []
            logging.info("collecting dot products")
            for pyu_device, batch_x_slice in batch_x.partitions.items():
                batch_o_slice = 0 if batch_o is None else batch_o.partitions[pyu_device]
                dot_products.append(
                    pyu_device(lambda x, w, o: x @ w + o)(
                        batch_x_slice,
                        fed_w.partitions[pyu_device],
                        batch_o_slice,
                    ).to(to_pyu)
                )

            logging.info("collect preds")
            preds.append(
                to_pyu(predict_based_on_dot_products, static_argnames=("link",))(
                    dot_products, bias, self.y_scale, link=self.link
                )
            )

        return to_pyu(lambda p: jnp.concatenate(p, axis=0))(preds)


def predict_based_on_dot_products(
    dot_products: List[np.ndarray],
    bias: float,
    y_scale: float,
    link: Linker,
):
    pred = np.sum(dot_products, axis=0) + bias
    return link.response(pred) * y_scale


def slice_x(x, beg, end) -> np.ndarray:
    return x[beg:end, 0]
