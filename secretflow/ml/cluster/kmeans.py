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
from sml.cluster import KMEANS

from secretflow.data.ndarray.ndarray import FedNdarray
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import wait
from secretflow.ml.base import _ModelBase


def _fit(x: jnp.ndarray, model) -> KMEANS:
    model = model.fit(x)
    return model


class KMeans(_ModelBase):
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
        n_clusters,
        init="k-means++",
        n_init=1,
        max_iter=300,
    ) -> None:
        spu_x = self._to_spu_dataset(x)
        logging.info(f"start to fit k-means model, n_clusters={n_clusters}")

        model = KMEANS(
            n_clusters=n_clusters,
            n_samples=x.shape[0],
            init=init,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.model = self.spu(_fit)(spu_x, model)
        wait([self.model])

    def predict(self, x: Union[FedNdarray, VDataFrame]) -> SPUObject:
        assert hasattr(self, 'model'), 'please fit model first'

        spu_x = self._to_spu_dataset(x)

        def _predict(x, model):
            return model.predict(x)

        return self.spu(_predict)(spu_x, self.model)
