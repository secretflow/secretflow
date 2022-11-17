#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


import abc
import logging
import math
from abc import abstractmethod
from typing import Tuple


class Criterion(abc.ABC):
    """Base class for split criterion"""

    @abstractmethod
    def split_gain(node_sum, left_node_sum, right_node_sum):
        pass


class XgboostCriterion(Criterion):
    """XgboostCriterion 分裂规则类
    Attributes:
        reg_lambda: L2 regularization term on weight
        reg_alpha: L1 regularization term on weight
        decimal: truncate parms
    """

    def __init__(
        self, reg_lambda: float = 0.1, reg_alpha: float = 0, decimal: int = 10
    ):
        assert (
            reg_lambda >= 0
        ), f" value {reg_lambda} for Parameter reg_lambda should be greater equal to 0"
        self.reg_lambda = reg_lambda  # l2 reg
        assert (
            reg_alpha >= 0
        ), f" value {reg_alpha} for Parameter reg_alpha should be greater equal to 0"
        self.reg_alpha = reg_alpha  # l1 reg
        self.decimal = decimal
        logging.debug(
            'splitter criterion setting done: l1 {}, l2 {}'.format(
                self.reg_alpha, self.reg_lambda
            )
        )

    @staticmethod
    def _g_alpha_cmp(gradient: float, reg_alpha: float) -> float:
        """L1 regularization on gradient
        Args:
            gradient: The value of the gradient
            reg_alpha: L1 regularization term
        """
        if gradient < -reg_alpha:
            return gradient + reg_alpha
        elif gradient > reg_alpha:
            return gradient - reg_alpha
        else:
            return 0

    def split_gain(
        self,
        node_sum: Tuple[float, float],
        left_node_sum: Tuple[float, float],
        right_node_sum: Tuple[float, float],
    ) -> float:
        """Calculate split gain
        Args:
            node_sum: After the split, Grad and Hess at this node
            left_node_sum:  After the split, Grad and Hess at the left split point
            right_node_sum: After the split, Grad and Hess at the right split point
        Returns:
            gain: Split gain of this split
        """
        sum_grad, sum_hess = node_sum
        left_node_sum_grad, left_node_sum_hess = left_node_sum
        right_node_sum_grad, right_node_sum_hess = right_node_sum
        gain = (
            self.node_gain(left_node_sum_grad, left_node_sum_hess)
            + self.node_gain(right_node_sum_grad, right_node_sum_hess)
            - self.node_gain(sum_grad, sum_hess)
        )
        return self.truncate(gain, decimal=self.decimal)

    @staticmethod
    def truncate(f, decimal=10):
        """Truncate control precision can reduce training time with early stop"""
        return math.floor(f * 10**decimal) / 10**decimal

    def node_gain(self, sum_grad: float, sum_hess: float) -> float:
        """Calculate node gain
        Args:
            sum_grad: Sum of gradient
            sum_hess: Sum of hessian
        Returns:
            Gain of this node
        """
        if sum_hess < 0:
            return 0.0
        sum_grad, sum_hess = self.truncate(
            sum_grad, decimal=self.decimal
        ), self.truncate(sum_hess, decimal=self.decimal)
        reg_grad = self._g_alpha_cmp(sum_grad, self.reg_alpha)
        gain = reg_grad * reg_grad / (sum_hess + self.reg_lambda)
        return self.truncate(gain, decimal=self.decimal)

    def node_weight(self, sum_grad: float, sum_hess: float) -> float:
        """Calculte node weight
        Args:
            sum_grad: Sum of gradient
            sum_hess: Sum of hessian
        Returns:
            Weight of this node
        """
        return self.truncate(
            -(self._g_alpha_cmp(sum_grad, self.reg_alpha))
            / (sum_hess + self.reg_lambda),
            decimal=self.decimal,
        )
