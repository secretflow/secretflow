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

from typing import List

import numpy as np
import jax.numpy as jnp

import secretflow as sf
from secretflow.data.vertical import VDataFrame
from secretflow.device import SPU
from secretflow.preprocessing.scaler import StandardScaler
from .core.utils import newton_matrix_inverse


class VIF:
    """
    Calculate variance inflation factor for vertical slice dataset
    by using secret sharing.

    see https://en.wikipedia.org/wiki/Variance_inflation_factor

    For large dataset(large than 10w samples & 200 features)
    Recommend use [Ring size: 128, Fxp: 40] options for SPU device.

    NOTICE:
    The analytical solution of matrix inversion in secret sharing is very expensive,
    so this method uses Newton iteration to find approximate solution.

    When there is multicollinearity in the input dataset, the XTX matrix is not full rank,
    and the analytical solution for the inverse of the XTX matrix does not exist.

    The VIF results of these linear correlational columns calculated by statsmodels are INF,
    indicating that the correlation is infinite.
    However, this method will get a large VIF value (>>1000) on these columns,
    which can also correctly reflect the strong correlation of these columns.

    When there are constant columns in the data, the VIF result calculated by statsmodels is NAN,
    and the result of this method is also a large VIF value (>> 1000),
    means these columns need to be removed before training.

    Therefore, although the results of this method cannot be completely consistent with statemodels
    that calculations in plain text, but they can still correctly reflect the correlation of the input data columns.

    Attributes:
        device: SPU Device
    """

    def __init__(self, device: SPU):
        self.spu_device = device

    def vif(self, vdata: VDataFrame, standardize: bool = True):
        """
        Attributes:

            vdata : VDataFrame
                vertical slice dataset.
            standardize: bool
                if need standardize dataset. dataset must be standardized
                please keep standardize=True, unless dataset is already standardized.
                standardize purpose:
                - reduce the result number of matrix xtx, avoid overflow in secret sharing.
                - after standardize, the variance is 1 and the mean is 0, which can simplify the calculation.

        """
        if standardize:
            scaler = StandardScaler()
            vdata = scaler.fit_transform(vdata)
        obj_list = [sf.to(self.spu_device, d.data) for d in vdata.partitions.values()]

        rows = vdata.shape[0]

        def spu_vif(objs: List[np.ndarray]):
            data = jnp.concatenate(objs, axis=1)
            xtx = jnp.matmul(data.transpose(), data)
            x_inv = newton_matrix_inverse(xtx)
            x_diagonal = jnp.diagonal(x_inv)
            return x_diagonal

        spu_obj = self.spu_device(spu_vif)(obj_list)
        x_diagonal = sf.reveal(spu_obj)
        return x_diagonal * (rows - 1)
