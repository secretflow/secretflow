# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import List, Tuple
import numpy as np
import jax.numpy as jnp
from scipy import stats

import secretflow as sf
from secretflow.utils.sigmoid import SigType
from secretflow.device import SPUObject, SPU
from secretflow.ml.linear import LinearModel, RegType, SSRegression
from secretflow.data.vertical import VDataFrame

from .core.utils import newton_matrix_inverse


# spu functions for Logistic PValue
def _hessian_matrix(x: np.ndarray, yhat: np.ndarray):
    """
    Hessian = X.T * A * X
       +---------------------------------------------+
       | y_hat(X1)[1-y_hat(x1)] 0    0   0   .. .  0 |
       |   0  y_hat(X2)[1-y_hat(x2)]                 |
    A =|   0                .                        |
       |                      .                      |
       |   0                  y_hat(Xm)[1-y_hat(xm)] |
       +---------------------------------------------+
    """
    A_dig = yhat * (1 - yhat)
    XAT = x * A_dig
    XTA = jnp.transpose(XAT)
    XTAX = jnp.matmul(XTA, x)
    return XTAX


def _z_square_value(x: List[np.ndarray], yhat: np.ndarray, w: np.ndarray):
    x = jnp.concatenate([*x, jnp.ones((x[0].shape[0], 1))], axis=1)
    assert x.shape[1] == w.shape[0], "weights' feature size != input x dataset's cols"
    assert x.shape[0] == yhat.shape[0], "x/y dataset not aligned"
    w = jnp.reshape(w, (w.shape[0],))
    yhat = jnp.reshape(yhat, (yhat.shape[0], 1))
    H = _hessian_matrix(x, yhat)
    H_inv = newton_matrix_inverse(H)
    H_inv_diag = jnp.diagonal(H_inv)
    return jnp.square(w) / H_inv_diag


# spu function for Linear PValue
def _t_square_value(
    x: List[np.ndarray], y: np.ndarray, yhat: np.ndarray, w: np.ndarray
):
    x = jnp.concatenate([*x, jnp.ones((x[0].shape[0], 1))], axis=1)
    assert x.shape[1] == w.shape[0], "weights' feature size != input x dataset's cols"
    assert x.shape[0] == yhat.shape[0], "x/y dataset not aligned"
    assert x.shape[0] == y.shape[0], "x/y dataset not aligned"
    w = jnp.reshape(w, (w.shape[0],))
    y = jnp.reshape(y, (y.shape[0], 1))
    yhat = jnp.reshape(yhat, (yhat.shape[0], 1))
    err = yhat - y
    sigma = jnp.matmul(jnp.transpose(err), err) / (x.shape[0] - x.shape[1] + 1)
    XTX = jnp.matmul(jnp.transpose(x), x)
    XTX_inv = newton_matrix_inverse(XTX)
    XTX_inv_diag = jnp.diagonal(XTX_inv)
    variance = XTX_inv_diag * sigma
    w_square = jnp.square(w)
    t_square = w_square / variance
    return t_square


class PVlaue:
    """
    Calculate P-Value for LR model training on vertical slice dataset by using secret sharing.

    more detail for P-Value:
    https://www.w3schools.com/datascience/ds_linear_regression_pvalue.asp

    For large dataset(large than 10w samples & 200 features)
    Recommend use [Ring size: 128, Fxp: 40] options for SPU device.

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

    def _linear_pvalue(
        self, x: VDataFrame, y: VDataFrame, yhat: SPUObject, weights: SPUObject
    ) -> np.ndarray:
        """
        computer pvalue for linear lr model

        Args:
            x: input dataset
            y: true label
            yhat: predict on x
            weights: model weights, last one is intercept/bias

        Return:
            PValue
        """
        x_shape = x.shape
        assert (
            x_shape[0] > x_shape[1]
        ), "num of samples must greater than num of features"
        x = self._prepare_dataset(x)
        y = self._prepare_dataset(y)
        assert len(y) == 1, "label should came from one party"
        y = y[0]
        spu_t = self.spu(_t_square_value)(x, y, yhat, weights)
        t_square = self._rectify_negative(sf.reveal(spu_t))
        t_values = np.sqrt(t_square)
        return 2 * (1 - stats.t(x_shape[0] - x_shape[1]).cdf(np.abs(t_values)))

    def _logistic_pvalue(
        self, x: VDataFrame, yhat: SPUObject, weights: SPUObject
    ) -> np.ndarray:
        """
        computer pvalue for logistic lr model

        Args:
            x: input dataset
            yhat: predict on x
            weights: model weights, last one is intercept/bias

        Return:
            PValue
        """
        x = self._prepare_dataset(x)
        spu_z = self.spu(_z_square_value)(x, yhat, weights)
        z_square = self._rectify_negative(sf.reveal(spu_z))
        wald_values = np.sqrt(z_square)
        return 2 * (1 - stats.norm.cdf(np.abs(wald_values)))

    def _rectify_negative(self, square: np.ndarray) -> np.ndarray:
        square = square.flatten()
        for idx in range(square.size):
            if square[idx] < 0:
                logging.warn(
                    f"square_mat has negative value {square[idx]} at feature {idx}"
                    "\nPlease check :\n1. if this feature is a const column or has strong correlation."
                    "\n2. if input model is converged."
                    "\n3. if dataset is same with the one used during training."
                    "\n4. if dataset is normalized or standardized."
                )
                square[idx] = 0
        return square

    def pvalues(self, x: VDataFrame, y: VDataFrame, model: LinearModel) -> np.ndarray:
        """
        computer pvalue for lr model

        Args:

            x : VDataFrame
                input dataset
            y : VDataFrame
                true label
            model : LinearModel
                lr model

        Return:
            PValue
        """
        assert isinstance(model.weights, SPUObject), (
            "Only support model fit by sslr/hesslr that "
            "training on vertical slice dataset."
        )
        assert model.weights.device == self.spu, "weights should saved in same spu"
        lr = SSRegression(self.spu)
        # hessian_matrix is very sensitive on yhat, use a expensive but more precision sig approximation.
        model.sig_type = SigType.MIX
        lr.load_model(model)
        yhat = lr.predict(x)
        if model.reg_type == RegType.Linear:
            return self._linear_pvalue(x, y, yhat, model.weights)
        else:
            return self._logistic_pvalue(x, yhat, model.weights)
