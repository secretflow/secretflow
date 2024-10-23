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
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from scipy import stats

import secretflow as sf
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.device import SPU, PYUObject, SPUObject, reveal
from secretflow.ml.linear.ss_glm.core.distribution import DistributionType, get_dist
from secretflow.ml.linear.ss_glm.core.link import LinkType, get_link
from secretflow.utils.blocked_ops import block_compute, cut_device_object, cut_vdata


def _compute_scale(
    spu: SPU,
    y: VDataFrame,
    yhat: SPUObject,
    power: float,
    sample_num_weighted: float,
    feature_rank: int,
    sample_weights: Union[None, PYUObject],
):
    y_blocks = cut_vdata(y, 100 * 10000, spu)
    yhat_blocks = cut_device_object(yhat, 100 * 10000, spu)
    if sample_weights is not None:
        sample_weights_blocks = cut_device_object(sample_weights, 100 * 10000, spu)
        blocks = zip(y_blocks, yhat_blocks, sample_weights_blocks)
    else:
        blocks = zip(y_blocks, yhat_blocks)

    def _rss(block):
        block_y, block_yhat = block
        block_yhat = block_yhat.reshape((-1, 1))
        block_y = block_y.reshape((-1, 1))
        return jnp.sum(jnp.square(block_y - block_yhat) / jnp.power(block_yhat, power))

    def _rss_weighted(block):
        block_y, block_yhat, block_sample_weights = block
        block_yhat = block_yhat.reshape((-1, 1))
        block_y = block_y.reshape((-1, 1))
        return jnp.sum(
            jnp.square(block_y - block_yhat)
            * block_sample_weights
            / jnp.power(block_yhat, power)
        )

    if sample_weights is not None:
        rss = reveal(block_compute(blocks, spu, _rss_weighted, lambda x, y: x + y))
    else:
        rss = reveal(block_compute(blocks, spu, _rss, lambda x, y: x + y))
    return rss / (sample_num_weighted - feature_rank - 1.0)


def rank_of_union_of_vectors(x):
    # I understand this is not right.
    # However, this is good enough for n >> p
    # If you find pvalue results is different from sklearn or statsmodel when n is small, say 300
    # it's normal.
    return x.shape[1]


def _hessian_matrix(
    spu: SPU,
    x: VDataFrame,
    y: VDataFrame,
    yhat: SPUObject,
    link: LinkType,
    dist: DistributionType,
    tweedie_power: float,
    row_number: int,
    y_scale: float,
    sample_weights: Union[PYUObject, None],
):
    if y_scale > 1:
        y_device = list(y.partitions.keys())[0]
        y.partitions[y_device] = partition(
            data=y_device(lambda y, scale: y / scale)(
                y.partitions[y_device].data, y_scale
            )
        )
        yhat = yhat.device(lambda y, scale: y / scale)(yhat, y_scale)

    x_shape = x.shape
    if sample_weights is not None:
        wnobs = sample_weights.device(lambda x: x.sum())(sample_weights)
        # may improve security later
        nf_model = reveal(wnobs)
    else:
        nf_model = x_shape[0]
    df_model = rank_of_union_of_vectors(x)

    if dist == DistributionType.Gamma:
        scale = _compute_scale(spu, y, yhat, 2, nf_model, df_model, sample_weights)
    elif dist == DistributionType.Tweedie:
        scale = _compute_scale(
            spu, y, yhat, tweedie_power, nf_model, df_model, sample_weights
        )
    else:
        scale = 1

    link = get_link(link)
    dist = get_dist(dist, scale, tweedie_power)

    def _h(x_y: List[np.ndarray]):
        x, yhat = x_y
        yhat = yhat.reshape((-1, 1))
        v = dist.variance(yhat)
        g_gradient = link.link_derivative(yhat)
        A_dig = 1 / dist.scale() / (v * g_gradient) / g_gradient
        XAT = x * A_dig
        XTA = jnp.transpose(XAT)
        XTAX = jnp.matmul(XTA, x)
        return XTAX

    def _h_w(x_y_sample_weights: List[np.ndarray]):
        x, yhat, sample_weights = x_y_sample_weights
        yhat = yhat.reshape((-1, 1))
        v = dist.variance(yhat)
        g_gradient = link.link_derivative(yhat)
        A_dig = 1 / dist.scale() / (v * g_gradient) / g_gradient
        A_dig *= sample_weights
        XAT = x * A_dig
        XTA = jnp.transpose(XAT)
        XTAX = jnp.matmul(XTA, x)
        return XTAX

    x_blocks = cut_vdata(x, row_number, spu, True)
    yhat_blocks = cut_device_object(yhat, row_number, spu)
    if sample_weights is not None:
        sample_weights_blocks = cut_device_object(sample_weights, row_number, spu)
        blocks = zip(x_blocks, yhat_blocks, sample_weights_blocks)
        return block_compute(blocks, spu, _h_w, lambda x, y: x + y)
    else:
        blocks = zip(x_blocks, yhat_blocks)
        return block_compute(blocks, spu, _h, lambda x, y: x + y)


def _xTx(
    spu: SPU,
    x: VDataFrame,
    row_number: int,
    sample_weights: Union[PYUObject, None] = None,
):
    def _xtx_w(x_w: List[np.ndarray]):
        x, w = x_w
        Xw = x * w
        XwT = jnp.transpose(Xw)
        XwTX = jnp.matmul(XwT, x)
        return XwTX

    def _xtx(x: List[np.ndarray]):
        return x.T @ x

    x_blocks = cut_vdata(x, row_number, spu, True)
    if sample_weights is not None:
        sample_weights_blocks = cut_device_object(sample_weights, row_number, spu)
        blocks = zip(x_blocks, sample_weights_blocks)
        return block_compute(blocks, spu, _xtx_w, lambda x, y: x + y)
    else:
        return block_compute(x_blocks, spu, _xtx, lambda x, y: x + y)


def _z_square_value(H_inv: np.ndarray, w: np.ndarray):
    assert (
        H_inv.shape[1] == w.shape[0]
    ), "weights' feature size != input x dataset's cols"
    w = jnp.reshape(w, (w.shape[0],))
    H_inv_diag = jnp.diagonal(H_inv)
    return jnp.square(w) / H_inv_diag


# spu function for Linear PValue
def _t_square_value(
    XTX_inv: np.ndarray, m: int, n: int, y: np.ndarray, yhat: np.ndarray, w: np.ndarray
):
    assert (
        XTX_inv.shape[1] == w.shape[0]
    ), f"weights' feature size {w.shape[0]}!= input x dataset's cols {XTX_inv.shape[1]}"
    w = jnp.reshape(w, (w.shape[0],))
    y = jnp.reshape(y, (y.shape[0], 1))
    yhat = jnp.reshape(yhat, (yhat.shape[0], 1))
    err = yhat - y
    sigma = jnp.matmul(jnp.transpose(err), err) / (m - n + 1)
    XTX_inv_diag = jnp.diagonal(XTX_inv)
    variance = XTX_inv_diag * sigma
    w_square = jnp.square(w)
    t_square = w_square / variance
    return t_square


class PValue:
    """
    Calculate P-Value for LR model training on vertical slice dataset by using secret sharing.

    more detail for P-Value:
    https://www.w3schools.com/datascience/ds_linear_regression_pvalue.asp

    For large dataset(large than 10w samples & 200 features)
    Recommend use [Ring size: 128, Fxp: 40] options for SPU device.

    For SS PVALUE Calculation, we mainly referenced stats model package. The only difference is
    that dof calculation is simplified.

    Attributes:

        device: SPU Device
    """

    def __init__(self, spu: SPU) -> None:
        self.spu = spu

    def _prepare_dataset(self, ds: VDataFrame) -> Tuple[SPUObject]:
        """
        check data setting and get total shape.

        Args:
            ds: input dataset

        Return:
            List of spu obj
        """
        shape = ds.shape
        assert shape[0] > 0 and shape[1] > 0, "input dataset is empty"

        return [ds.partitions[pyu].data.to(self.spu) for pyu in ds.partitions]

    def _rectify_negative(self, square: np.ndarray) -> np.ndarray:
        square = square.flatten()
        for idx in range(square.size):
            if square[idx] < 0:
                logging.warning(
                    f"square_mat has negative value {square[idx]} at feature {idx}"
                    "\nPlease check :\n1. if this feature is a const column or has strong correlation."
                    "\n2. if input model is converged."
                    "\n3. if dataset is same with the one used during training."
                    "\n4. if dataset is normalized or standardized."
                )
                square[idx] = 0
        return square

    def _pre_check_sample_weights(
        self,
        x: VDataFrame,
        sample_weights: Union[None, PYUObject, VDataFrame],
    ) -> Union[None, PYUObject]:
        # reduce the case of VDataFrame to PYUObject, inside which is a np array
        if isinstance(sample_weights, VDataFrame):
            if x.shape[0] != sample_weights.shape[0]:
                raise ValueError(
                    "sample_weights should have same row number with x, but got x shape {} and sample weights shape {}".format(
                        x.shape, sample_weights.shape
                    )
                )
            if sample_weights.shape[1] != 1:
                raise ValueError(
                    "sample_weights should have shape (x, 1), but got {}".format(
                        sample_weights.shape
                    )
                )
            sample_weights = list(sample_weights.partitions.values())[0].data
            sample_weights = sample_weights.device(lambda x: x.values)(sample_weights)

        if sample_weights is not None:
            sample_weights = sample_weights.device(lambda x: x.reshape(-1, 1))(
                sample_weights
            )
        # the result sample weights is a PYUObject, inside which is a numpy array (n, 1)
        return sample_weights

    def _pre_check(
        self,
        x: VDataFrame,
        y: VDataFrame,
        yhat: SPUObject,
        weights: SPUObject,
        infeed_elements_limit: int,
    ):
        assert x.shape[0] == y.shape[0], "x/y dataset not aligned"
        assert isinstance(yhat, SPUObject)
        assert isinstance(weights, SPUObject), (
            "Only support model fit by sslr/hesslr/glm that "
            "training on vertical slice dataset."
        )
        assert yhat.device == self.spu
        assert weights.device == self.spu
        x_shape = x.shape
        assert (
            x_shape[0] > x_shape[1]
        ), "num of samples must greater than num of features"
        assert x.shape[0] == reveal(
            self.spu(lambda yhat: yhat.shape[0])(yhat)
        ), "x/y dataset not aligned"

        return max([math.ceil(infeed_elements_limit / x.shape[1]), 1])

    def t_statistic_p_value(
        self,
        x: VDataFrame,
        y: VDataFrame,
        yhat: SPUObject,
        weights: SPUObject,
        infeed_elements_limit: int = 20000000,
    ) -> np.ndarray:
        """
        compute pvalue by t-statistic, use for linear regression with normal distribution assumption.

        Args:

            x : VDataFrame
                input dataset
            y : VDataFrame
                true label
            yhat : SPUObject
                predicted label
            weights : SPUObject
                features' weight
            sample_weights: Union[None, PYUObject]
                sample weights
        Return:
            PValue
        """
        # TODO: complete sample weights for t statistic pvalue later
        # compute scale similar to z statistics
        # expose api after linear model supported weights
        sample_weights = None
        row_number = self._pre_check(x, y, yhat, weights, infeed_elements_limit)
        if x.shape[0] <= x.shape[1] + 1:
            raise ValueError(
                "Security check not passed, if num of samples is less than num of features + 1, then label holder may infer all data."
            )
        y_device = list(y.partitions.keys())[0]
        xtx = _xTx(self.spu, x, row_number, sample_weights)
        y = self._prepare_dataset(y)
        assert len(y) == 1, "label should came from one party"
        y = y[0]
        x_shape = x.shape

        xtx_inv = y_device(lambda x: np.linalg.pinv(x))(xtx.to(y_device)).to(self.spu)

        if sample_weights is not None:
            wnobs = sample_weights.device(lambda x: x.sum())(sample_weights)
            # may improve security later
            nf_model = reveal(wnobs)
        else:
            nf_model = x_shape[0]
        df_model = rank_of_union_of_vectors(x)
        spu_t = self.spu(_t_square_value)(xtx_inv, nf_model, df_model, y, yhat, weights)
        t_square = self._rectify_negative(sf.reveal(spu_t))
        t_values = np.sqrt(t_square)
        return 2 * (1 - stats.t(nf_model - df_model - 1).cdf(np.abs(t_values)))

    def z_statistic_p_value(
        self,
        x: VDataFrame,
        y: VDataFrame,
        yhat: SPUObject,
        weights: SPUObject,
        link: LinkType,
        dist: DistributionType,
        tweedie_power: float = 1,
        y_scale: float = 1,
        infeed_elements_limit: int = 20000000,
        sample_weights=None,
    ) -> np.ndarray:
        from secretflow.ml.linear import LinearModel, RegType, SSRegression

        """
        compute pvalue by z-statistic, use for general linear regression with Non-normal distribution assumption.

        Args:

            x : VDataFrame
                input dataset
            y : VDataFrame
                true label
            yhat : SPUObject
                predicted label
            weights : SPUObject
                features' weight
            link : LinkType
                link function type
            dist : DistributionType
                label distribution type
            tweedie_power : float
                Specify power for tweedie distribution
            y_scale: float
                Specify y label scaling sparse used in glm training
            sample_weights: Union[None, PYUObject, VDataFrame]
                Specify the weight of each sample, should have shape (x, 1)

        WARNING:
        calculating pvalues will reveal model weight to label holder. Please make sure you have
        enough security guarantee. Revealing model weight to label holder may lead to privacy leakage.

        Return:
            PValue
        """
        if x.shape[0] <= x.shape[1] + 1:
            raise ValueError(
                "Security check not passed, if num of samples is less than num of features + 1, then label holder may infer all data."
            )
        row_number = self._pre_check(x, y, yhat, weights, infeed_elements_limit)
        sample_weights = self._pre_check_sample_weights(x, sample_weights)
        H = _hessian_matrix(
            self.spu,
            x,
            y,
            yhat,
            link,
            dist,
            tweedie_power,
            row_number,
            y_scale,
            sample_weights,
        )

        y_device = list(y.partitions.keys())[0]
        H_inv = y_device(lambda x: np.linalg.pinv(x))(H.to(y_device)).to(self.spu)

        spu_z = self.spu(_z_square_value)(H_inv, weights)
        z_square = self._rectify_negative(sf.reveal(spu_z))
        wald_values = np.sqrt(z_square)
        return 2 * (1 - stats.norm.cdf(np.abs(wald_values)))
