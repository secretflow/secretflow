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


"""keras global evaluation metrics

"""
from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
import logging


# The reason we just do not inherit or combine tensorflow metrics
# is tensorflow metrics are un-serializable but we need send they from worker to server.


class Metric(ABC):
    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def __radd__(self, other):
        pass

    @abstractmethod
    def __add__(self, other):
        pass


@dataclass
class Default(Metric):
    name: str
    total: float
    count: float

    def __add__(self, other):
        assert self.name == other.name
        total = self.total + other.total
        count = self.count + other.count
        return Default(self.name, total, count)

    def __radd__(self, other):
        assert other == 0
        return Default(self.name, self.total, self.count)

    def result(self):
        logging.warn(
            "Please pay attention to local metrics, global only do naive aggregation "
        )
        import tensorflow as tf

        metric = tf.keras.metrics.Mean()
        metric.total = self.total
        metric.count = self.count
        return metric.result()


@dataclass
class Mean(Metric):
    """keras.metrics.Mean on fede

    Attributes:
        total: sum of metrics
        count: num of samples
    """

    name: str
    total: float
    count: float

    def __radd__(self, other):
        assert other == 0
        return Mean(self.name, self.total, self.count)

    def __add__(self, other: 'Mean'):
        assert self.name == other.name
        total = self.total + other.total
        count = self.count + other.count
        return Mean(self.name, total, count)

    def result(self):
        import tensorflow as tf

        metric = tf.keras.metrics.Mean()
        metric.total = self.total
        metric.count = self.count
        return metric.result()


class AUC(Metric):
    """Federated keras.metrics.AUC

    Attributes:
        thresholds: threshold of buckets. same to tf.keras.metrics.AUC,must contain 0 and 1.
        true_positives: num samples of true positive.
        true_negatives: num samples of true negative.
        false_positives: num samples of false positive.
        false_negatives: num samples of false negative.
        curve: type of AUC curve, same to 'tf.keras.metrics.AUC', it can be 'ROC' or 'PR'.
    """

    def __init__(
        self,
        name: str,
        thresholds: List[float],
        true_positives: List[float],
        true_negatives: List[float],
        false_positives: List[float],
        false_negatives: List[float],
        curve=None,
    ):
        self.name = name
        self.thresholds = thresholds
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        if curve is not None:
            self.curve = curve
        else:
            from tensorflow.python.keras.utils.metrics_utils import AUCCurve

            self.curve: AUCCurve = AUCCurve.ROC

    def __radd__(self, other):
        assert other == 0
        return AUC(
            self.name,
            self.thresholds,
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
            self.curve,
        )

    def __add__(self, other: 'AUC'):
        assert self.name == other.name
        assert self.curve == other.curve, f'Curves are different!'
        assert len(self.thresholds) == len(other.thresholds) and all(
            i == j for i, j in zip(self.thresholds, other.thresholds)
        ), f'Thresholds are different!'
        true_positives = self.true_positives + other.true_positives
        true_negatives = self.true_negatives + other.true_negatives
        false_positives = self.false_positives + other.false_positives
        false_negatives = self.false_negatives + other.false_negatives
        return AUC(
            self.name,
            self.thresholds,
            true_positives,
            true_negatives,
            false_positives,
            false_negatives,
            self.curve,
        )

    def result(self):
        import tensorflow as tf

        # 由于tf.keras.metrics.AUC会默认给thresholds添加{-epsilon, 1+epsilon}两个边界值，因此这里需要去掉两个边界点。
        metric = tf.keras.metrics.AUC(
            thresholds=self.thresholds[1:-1], curve=self.curve
        )
        metric.true_positives = self.true_positives
        metric.true_negatives = self.true_negatives
        metric.false_positives = self.false_positives
        metric.false_negatives = self.false_negatives
        return metric.result()


@dataclass
class Precision(Metric):
    """Federated keras.metrics.Precision

    Attributes:
        thresholds: value of threshold, float or list, in [0, 1].
        true_positives: num samples of true positive
        false_positives: num samples of false positive
    """

    name: str
    thresholds: float
    true_positives: float
    false_positives: float

    def __radd__(self, other):
        assert other == 0
        return Precision(
            self.name, self.thresholds, self.true_positives, self.false_positives
        )

    def __add__(self, other: 'Precision'):
        assert self.name == other.name
        thresholds = self.thresholds
        true_positives = self.true_positives + other.true_positives
        false_positives = self.false_positives + other.false_positives
        return Precision(self.name, thresholds, true_positives, false_positives)

    def result(self):
        import tensorflow as tf

        metric = tf.keras.metrics.Precision()
        metric.thresholds = self.thresholds
        metric.true_positives = self.true_positives
        metric.false_positives = self.false_positives
        return metric.result()


@dataclass
class Recall(Metric):
    """Federated keras.metrics.Recall

    Attributes:
        thresholds: value of threshold, float or list, in [0, 1].
        true_positives: num samples of true positive
        false_negatives: num samples of false negative

    """

    name: str
    thresholds: float
    true_positives: float
    false_negatives: float

    def __radd__(self, other):
        assert other == 0
        return Recall(
            self.name, self.thresholds, self.true_positives, self.false_negatives
        )

    def __add__(self, other: 'Recall'):
        assert self.name == other.name
        thresholds = self.thresholds
        true_positives = self.true_positives + other.true_positives
        false_negatives = self.false_negatives + other.false_negatives
        return Recall(self.name, thresholds, true_positives, false_negatives)

    def result(self):
        import tensorflow as tf

        metric = tf.keras.metrics.Recall()
        metric.thresholds = self.thresholds
        metric.true_positives = self.true_positives
        metric.false_negatives = self.false_negatives
        return metric.result()


def aggregate_metrics(local_metrics: List[List]) -> List:
    """Aggregate Model metrics values of each party and calculate global metrics.

    Args:
        local_metrics: Model metrics values in this party.

    Returns:
        A list of aggregations of each party metrics.
    """
    return [sum(metrics) for metrics in zip(*local_metrics)]
