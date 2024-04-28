# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Union

import jax.numpy as jnp

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, PYUObject, SPU, SPUObject
from secretflow.ml.boost.core.data_preprocess import prepare_dataset

from .core import node_split as split_fn
from .core.node_split import RegType
from .core.tree_worker import XgbTreeWorker as Worker


class XgbModel:
    """
    SS Xgb Model & predict.
    """

    def __init__(self, spu: SPU, objective: RegType, base: float) -> None:
        self.spu = spu
        self.objective = objective
        self.base = base
        # List[Dict[PYU, PYUObject of XgbTree]], owned by pyu, only knows split value if feature belong to this pyu.
        self.trees = list()
        # List[SPUObject of np.array], owned by spu and not reveal to any one
        self.weights = list()

    def _tree_pred(self, tree: Dict[PYU, PYUObject], weight: SPUObject) -> SPUObject:
        assert len(tree) == len(self.x)

        weight_selects = list()
        for worker in self.workers:
            device = worker.device
            assert device in tree
            s = worker.predict_weight_select(self.x[device].data, tree[device])
            weight_selects.append(s.to(self.spu))

        pred = self.spu(split_fn.predict_tree_weight)(weight_selects, weight)

        return pred

    def predict(
        self,
        dtrain: Union[FedNdarray, VDataFrame],
        to_pyu: PYU = None,
    ) -> Union[SPUObject, FedNdarray]:
        """
        predict on dtrain with this model.

        Args:

            dtrain : [FedNdarray, VDataFrame]
                vertical split dataset.

            to: the prediction initiator
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                otherwise, keep predict result in secret and save as SPUObject.

        Return:
            Pred values store in spu object or FedNdarray.
        """
        if len(self.trees) == 0:
            return None
        x, _ = prepare_dataset(dtrain)
        assert len(x.partitions) == len(
            self.trees[0]
        ), f"{len(x.partitions)}, {self.trees[0]}"
        self.workers = [Worker(0, device=pyu) for pyu in x.partitions]
        self.x = x.partitions
        pred = 0
        for idx in range(len(self.trees)):
            pred = self.spu(lambda x, y: jnp.add(x, y))(
                self._tree_pred(self.trees[idx], self.weights[idx]), pred
            )

        pred = self.spu(lambda x, y: jnp.add(x, y).reshape(-1, 1))(pred, self.base)

        if self.objective == RegType.Logistic:
            pred = self.spu(split_fn.sigmoid)(pred)

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

    def get_objective(self):
        return self.objective

    def get_trees(self):
        return self.trees

    def get_weights(self):
        return self.weights
