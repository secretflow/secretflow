# Copyright 2023 Ant Group Co., Ltd.
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


# supports merging a distributed model into a single party model, and thereby supports standalone deployment.
# WARNING: DO NOT USE THIS, UNLESS YOU KNOW EXACTLY WHAT THIS IS DOING.
# THIS FEATURE IS NOT SAFE. ALL INFO IN THE MODEL IS REVEALED TO OWNER PARTY.
# MOST IMPORTANTLY, BASICALLY ALL LABEL INFORMATION IS REVEALED TO OWNER PARTY.

import logging
from typing import Dict, List

import jax.numpy as jnp
import numpy as np

from secretflow.device import PYU, PYUObject

from .complete_tree import CompleteTree
from .complete_tree import from_dict as complete_tree_from_dict
from .complete_tree import from_distributed_tree
from .core.params import RegType
from .core.pure_numpy_ops.pred import sigmoid
from .model import SgbModel


class CompleteSgbModel:
    """
    Complete Sgboost Model & predict. It is a  set of standalone trees in essence.
    WARNING: DO NOT USE THIS, UNLESS YOU KNOW EXACTLY WHAT THIS IS DOING.
    THIS FEATURE IS NOT SAFE. ALL INFO IN THE MODEL IS REVEALED TO OWNER PARTY.
    MOST IMPORTANTLY, BASICALLY ALL LABEL INFORMATION IS REVEALED TO OWNER PARTY.
    """

    def __init__(self) -> None:
        """
        Args:
            objective: RegType, specifies doing logistic regression or regression
            base: float
        """
        self.objective = None
        self.base = None
        self.trees: List[CompleteTree] = []

    def predict(
        self,
        x: np.array,
    ) -> np.array:
        """
        predict on x with this model.

        Example:
            # suppose model and x are pyu objects from the same party, then
            pred_pyu_object = party_pyu(lambda model, x: model.predict(x))(model, x)
        """
        preds = []
        for tree in self.trees:
            pred = tree.predict(x)
            preds.append(pred)

        pred = (jnp.sum(jnp.concatenate(preds, axis=1), axis=1) + self.base).reshape(
            -1, 1
        )

        if self.objective == RegType.Logistic:
            pred = sigmoid(pred)

        return pred

    def to_dict(self):
        return {
            "objective": self.objective.value,
            "base": self.base,
            "trees": [tree.to_dict() for tree in self.trees],
        }


def from_sgb_model_parts(
    objective: RegType, base: float, trees: List[CompleteTree]
) -> CompleteSgbModel:
    model = CompleteSgbModel()
    model.objective = objective
    model.base = base
    model.trees: List[CompleteTree] = trees
    return model


def from_sgb_model(model_distributed: SgbModel, party_pyu: PYU) -> PYUObject:
    """
    build a complete model for party_pyu with model_distributed. Note all model information will be passed to party_pyu.

    MOST IMPORTANTLY, BASICALLY ALL LABEL INFORMATION IS REVEALED TO OWNER PARTY.

    Example:
        complete_model_pyu_object = build_complete_model(model_distributed, alice)
        model_dict_pyu_object = complete_model_pyu_object.device(lambda model: model.to_dict())(complete_model_pyu_object)
        complete_model_pyu_object.device(lambda x: json.dump(x, open("model.json", "w"))(model_dict_pyu_object)
    """
    complete_trees_pyuobject = [
        from_distributed_tree(party_pyu, tree) for tree in model_distributed.trees
    ]
    logging.warning(
        f"ALL MODEL INFO IS REVEALED TO PARTY {party_pyu.party} AND THIS PARTY WILL BASICALLY PREDICT ALL LABELS. THIS PARTY CAN DEPLOY MODEL WITHOUT OTHER PARTIES."
    )
    return party_pyu(from_sgb_model_parts)(
        model_distributed.objective, model_distributed.base, complete_trees_pyuobject
    )


def from_dict(dict: Dict) -> CompleteSgbModel:
    """load a complete model from a dict.

    Examples:
        model = from_dict(json.load(open("model.json", "r")))

        # or if with pyu
        path = "model.json"
        model_pyu_object = pyu(lambda x: from_dict(json.load(open(x, "r"))))(path)
    """
    model = CompleteSgbModel()
    model.objective = RegType(dict["objective"])
    model.base = dict["base"]
    model.trees = [complete_tree_from_dict(tree) for tree in dict["trees"]]
    return model
