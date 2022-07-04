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

import jax.numpy as jnp

import secretflow as sf
from secretflow.data.vertical import VDataFrame
from secretflow.device import SPU, SPUObject
from secretflow.preprocessing.scaler import StandardScaler


class VertPearsonR:
    """
    Calculate pearson product-moment correlation coefficient for vertical slice dataset.
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
            vdata: vertical slice dataset.
            standardize: if need standardize dataset
        """
        if standardize:
            scaler = StandardScaler()
            vdata = scaler.fit_transform(vdata)
        obj_list = [sf.to(self.spu_device, d.data) for d in vdata.partitions.values()]

        def spu_xtx(objs: List[SPUObject]):
            data = jnp.concatenate(objs, axis=1)
            return jnp.matmul(data.transpose(), data), data.shape[0]

        spu_obj = self.spu_device(spu_xtx)(obj_list)
        xtx, rows = sf.reveal(spu_obj)
        return xtx / (rows - 1)
