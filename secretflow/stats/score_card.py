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

from typing import Union

import math

import numpy as np

from secretflow.data.vertical import VDataFrame
from secretflow.data.horizontal import HDataFrame
from secretflow.data import FedNdarray


class ScoreCard:
    """
    The component provides a mapping procedure from binary regression's probability value to an integer range score.

    The mapping process is as follows:
        odds = pred / (1 - pred)
        score = offset + factor * log(odds)

    The offset and factor in the formula come from the user's settings. Usually users do not directly give offset and factor, but give three constraint parameters:
        scaled_value: a score baseline
        odd_base: the odds value at given score baseline
        pdo: how many scores are needed to double odds

    The offset and factor can be solved using these three constraint parameters:
        factor = pdo / log(2)
        offset = scaled_value - (factor * log(odd_base))

    Attributes:

        odd_base / scaled_value / pdo: see above

        max_score: up limit for score

        min_score: down limit for score

        bad_label_value:  which label represents the negative sample

    """

    def __init__(
        self,
        odd_base: float,
        scaled_value: float,
        pdo: float,
        max_score: int = 1000,
        min_score: int = 0,
        bad_label_value: int = 0,
    ):
        assert odd_base > 0, f"odd_base should be positive, got {odd_base}"
        assert scaled_value > 0, f"scaled_value should be positive, got {scaled_value}"
        assert pdo > 0, f"pdo should be positive, got {pdo}"
        assert (
            max_score >= 0 and max_score > scaled_value
        ), f"max_score should bigger than 0 and scaled_value, got {max_score}"
        assert (
            min_score >= 0 and min_score < scaled_value and scaled_value < max_score
        ), f"min_score should bigger than 0 but less than scaled_value and max_score, got {min_score}"
        assert bad_label_value in [
            0,
            1,
        ], f"bad_label_value should be 0 or 1, got {bad_label_value}"

        self.factor = pdo / math.log(2)
        self.offset = scaled_value - self.factor * math.log(odd_base)
        self.max_score = max_score
        self.min_score = min_score
        self.bad_label_value = bad_label_value

    def transform(self, pred: Union[FedNdarray, VDataFrame, HDataFrame]) -> FedNdarray:
        """
        computer pvalue for lr model

        Args:

            pred : Union[FedNdarray, VDataFrame, HDataFrame]
                predicted probability from binary regression

        Return:

            mapped scores.
        """

        assert isinstance(
            pred, (FedNdarray, VDataFrame, HDataFrame)
        ), "pred should be FedNdarray or VDataFrame or HDataFrame"
        pred = pred if isinstance(pred, FedNdarray) else pred.values
        shape = pred.shape
        assert len(shape) == 1 or shape[1] == 1, "pred should be list or 1D array"

        def score_transform(pred: np.ndarray):
            assert (pred >= 0).all() and (
                pred <= 1
            ).all(), f"pred should in [0, 1], but got max pred {pred.max()} and min pred {pred.min()}"
            if self.bad_label_value == 1:
                score = self.offset - self.factor * np.log(pred / (1 - pred))
            else:
                score = self.offset + self.factor * np.log(pred / (1 - pred))

            score = np.select(
                [score > self.max_score, score < self.min_score],
                [self.max_score, self.min_score],
                score,
            )
            return score

        return FedNdarray(
            partitions={
                d: d(score_transform)(pred.partitions[d]) for d in pred.partitions
            },
            partition_way=pred.partition_way,
        )
