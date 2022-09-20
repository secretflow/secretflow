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

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from secretflow.utils.errors import InvalidArgumentError

LEGAL_OBJ_FUNCTION = [
    "binary:logistic",
    "reg:logistic",
    "multi:softmax",
    "multi:softprob",
    "reg:squarederror",
]


class LossFunction(object):
    """Inner define for loss functions

    Attributes:
        obj_name: Name of loss function in
            ["binary:logistic",# logistic regression
            "reg:logistic", # logistic regression for binary classification, output probability
            "multi:softmax", # logistic regression for binary classification, output score before logistic transformation
            "multi:softprob", # logistic regression for binary classification, output probability
            "reg:squarederror" # for multi label classification
            ]
    """

    def __init__(self, obj_name: str):
        self.obj_name = obj_name
        if not self._check_legal(obj_name):
            raise InvalidArgumentError("Illegal loss function params")

    @staticmethod
    def _check_legal(obj_name: str) -> bool:
        if obj_name in LEGAL_OBJ_FUNCTION:
            return True
        else:
            return False

    def obj_function(self):
        if self.obj_name == "binary:logistic" or self.obj_name == "reg:logistic":
            return self._reg_logistic
        if self.obj_name == "multi:softmax" or self.obj_name == "multi:softprob":
            return self._softmaxobj
        if self.obj_name == "reg:squarederror":
            return self._reg_squared

    def _reg_logistic(self, preds, dtrain):
        """logistic objective.
        Args:
            preds: (N, 1) array, N = #data, K = #classes.
            dtrain: DMatrix object with training data.

        Returns:
            grad: N*1 array with gradient values.
            hess: N*1 array with second-order gradient values.
        """
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # transform raw leaf weight
        grad = (preds - labels).astype(np.float64)
        hess = (preds * (1.0 - preds)).astype(np.float64)
        return grad, hess

    def _reg_squared(self, preds, dtrain):
        """Squared loss objective.
        Args:
            preds: (N, 1) array, N = #data, K = #classes.
            dtrain: DMatrix object with training data.

        Returns:
            grad: N*1 array with gradient values.
            hess: N*1 array with second-order gradient values.
        """
        labels = dtrain.get_label()
        grad = (-2 * (labels - preds)).astype(np.float64)
        hess = (np.ones_like(labels) * 2).astype(np.float64)
        return grad, hess

    def _softmaxobj(self, preds, dtrain):
        """Softmax objective.
        Args:
            preds: (N, K) array, N = #data, K = #classes.
            dtrain: DMatrix object with training data.

        Returns:
            grad: N*K array with gradient values.
            hess: N*K array with second-order gradient values.
        """

        def _softmax(x):
            '''Softmax function with x as input vector.'''
            e = np.exp(x)
            return e / np.sum(e, axis=1).reshape(-1, 1)

        # Label is a vector of class indices for each input example
        labels = dtrain.get_label()
        # When objective=softprob, preds has shape (N, K)
        labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))
        prob = _softmax(preds)
        grad = (prob - labels).astype(np.float64)
        hess = (2.0 * prob * (1.0 - prob)).astype(np.float64)
        # Return as n-m metrics
        return grad, hess
