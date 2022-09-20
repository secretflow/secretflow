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


class PearsonR:
    """
    Calculate pearson product-moment correlation coefficient for vertical slice dataset
    by using secret sharing.

    more detail for PearsonR:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    For large dataset(large than 10w samples & 200 features)
    Recommend use [Ring size: 128, Fxp: 40] options for SPU device.

    Attributes:

        device: SPU Device
    """

    def __init__(self, device: SPU):
        self.spu_device = device

    def pearsonr(self, vdata: VDataFrame, standardize: bool = True):
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

        def spu_xtx(objs: List[np.ndarray]):
            data = jnp.concatenate(objs, axis=1)
            return jnp.matmul(data.transpose(), data)

        spu_obj = self.spu_device(spu_xtx)(obj_list)
        xtx = sf.reveal(spu_obj)
        return xtx / (rows - 1)
