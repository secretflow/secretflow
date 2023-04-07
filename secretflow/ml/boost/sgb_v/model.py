# Copyright 2023 Ant Group Co., Ltd.
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


from typing import Union

import jax.numpy as jnp

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, PYU, PYUObject

from .core.label_holder import RegType
from .core.preprocessing import prepare_dataset
from .core.pure_numpy_ops.pred import sigmoid
from .core.distributed_tree.distributed_tree import DistributedTree


class SgbModel:
    """
    Sgboost Model & predict. It is a distributed tree in essence.
    """

    def __init__(
        self, heu: HEU, label_holder: PYU, objective: RegType, base: float
    ) -> None:
        """
        Args:
            heu: HEU device, secret key keeper must belong to label_holder's party
            label_holder: PYU device, label holder's PYU device.
            objective: RegType, specifies doing logistic regression or regression
            base: float
        """
        assert heu.sk_keeper_name() == label_holder.party, (
            f"HEU sk keeper party {heu.sk_keeper_name()} "
            "mismatch with label_holder device's party {label_holder.party}"
        )
        self.heu = heu
        self.label_holder = label_holder
        self.objective = objective
        self.base = base
        # List[DistributedTree]
        self.trees = list()
        # TODO how to ser/der ?

    def _insert_distributed_tree(self, tree: DistributedTree):
        self.trees.append(tree)

    def predict(
        self,
        dtrain: Union[FedNdarray, VDataFrame],
        to_pyu: PYU = None,
    ) -> Union[PYUObject, FedNdarray]:
        """
        predict on dtrain with this model.

        Args:

            dtrain : [FedNdarray, VDataFrame]
                vertical split dataset.

            to: the prediction initiator
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                otherwise, keep predict result in plaintext and save as PYUObject in label_holder device.

        Return:
            Pred values store in pyu object or FedNdarray.
        """
        if len(self.trees) == 0:
            return None
        x, _ = prepare_dataset(dtrain)
        x = x.partitions
        preds = []
        for tree in self.trees:
            pred = tree.predict(x)
            preds.append(pred)

        pred = self.label_holder(
            lambda ps, base: (
                jnp.sum(jnp.concatenate(ps, axis=0), axis=0) + base
            ).reshape(-1, 1)
        )(preds, self.base)

        if self.objective == RegType.Logistic:
            pred = self.label_holder(sigmoid)(pred)

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
