# Copyright 2025 Ant Group Co., Ltd.
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
from typing import Union

import jax.numpy as jnp
from sml.gaussian_process import GaussianProcessClassifier

from secretflow.data.ndarray.ndarray import FedNdarray
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import wait
from secretflow.ml.base import _ModelBase


def _fit(x: jnp.ndarray, y: jnp.ndarray, model) -> GaussianProcessClassifier:
    model = model.fit(x, y)
    return model


class GPC(_ModelBase):
    def __init__(self, spu: SPU):
        super().__init__(spu)

    def _to_spu_dataset(self, x: Union[FedNdarray, VDataFrame]) -> SPUObject:
        x, _ = self._prepare_dataset(x)
        return self.spu(self._concatenate, static_argnames=('axis'))(
            self._to_spu(x),
            axis=1,
        )

    def fit(
        self,
        x: Union[FedNdarray, VDataFrame],
        y: Union[FedNdarray, VDataFrame],
        max_iter_predict=20,
        n_classes=2,
        poss='sigmoid',  # only support sigmoid now
        multi_class="one_vs_rest",  # only support one_vs_rest now
        kernel=None,  # only support RBF now
    ) -> None:
        if len(y.shape) == 2:
            if y.shape[1] != 1:
                raise ValueError('y should be 1D array')
        spu_x = self._to_spu_dataset(x)
        spu_y = self._to_spu(y)[0]

        def adjust_label_shape(y: jnp.ndarray):
            y = y.reshape(-1)
            return y

        spu_y = self.spu(adjust_label_shape)(spu_y)

        logging.info(f'fit gpc model..., x_shape:{x.shape} y_shape:{y.shape}')

        model = GaussianProcessClassifier(
            kernel=kernel,
            max_iter_predict=max_iter_predict,
            n_classes=n_classes,
            poss=poss,
            multi_class=multi_class,
        )
        self.model = self.spu(_fit)(spu_x, spu_y, model)
        wait([self.model])

    def predict(self, x: Union[FedNdarray, VDataFrame]) -> SPUObject:
        assert hasattr(self, 'model'), 'please fit model first'

        spu_x = self._to_spu_dataset(x)

        def _predict(x, model):
            return model.predict(x)

        return self.spu(_predict)(spu_x, self.model)
