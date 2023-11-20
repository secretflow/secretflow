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

from typing import Tuple

import numpy as np
import jax.numpy as jnp


from ....core.params import RegType
from ....core.pure_numpy_ops.grad import (
    compute_gh_linear,
    compute_gh_logistic,
    compute_relative_scaling_factor,
    compute_sum_abs,
    scale,
)


class LossComputerActor:
    """Actual Actor that computes loss, gradients and hessians"""

    def __init__(self) -> None:
        self.abs_g_cache = None
        self.abs_h_cache = None
        self.last_abs_g_cache = None

    def compute_gh(
        self, y: np.ndarray, pred: np.ndarray, obj: RegType
    ) -> Tuple[np.ndarray, np.ndarray]:
        if obj == RegType.Linear:
            g, h = compute_gh_linear(y, pred)
        elif obj == RegType.Logistic:
            g, h = compute_gh_logistic(y, pred)
        else:
            raise f"unknown objective {obj}"
        return g, h

    def compute_abs_sums(self, g: np.ndarray, h: np.ndarray):
        abs_g_sum = compute_sum_abs(g)
        abs_h_sum = compute_sum_abs(h)

        self.last_abs_g_cache = self.abs_g_cache
        self.abs_g_cache = abs_g_sum
        self.abs_h_cache = abs_h_sum
        return

    def compute_scales(self, scaling: float):
        abs_g_sum = self.abs_g_cache
        abs_h_sum = self.abs_h_cache
        self.g_scale = compute_relative_scaling_factor(abs_g_sum, scaling)
        self.h_scale = compute_relative_scaling_factor(abs_h_sum, scaling)
        return

    def check_abs_g_sum_early_stop(self, threshold: float) -> bool:
        # early stopping happened, and this is known by all parties
        abs_sum = self.abs_g_cache
        if abs_sum is None:
            return False
        return abs_sum <= threshold

    def check_abs_g_sum_change_ratio_early_stop(self, threshold: float) -> bool:
        old = self.last_abs_g_cache
        current_abs_sum = self.abs_g_cache
        if old is None or current_abs_sum is None:
            return False
        return delta_ratio(old, current_abs_sum) <= threshold

    def check_early_stop(
        self, abs_sum_threshold: float, abs_sum_change_ratio_threshold: float
    ) -> bool:
        g_sum_es = self.check_abs_g_sum_early_stop(abs_sum_threshold)
        g_sum_delta_es = self.check_abs_g_sum_change_ratio_early_stop(
            abs_sum_change_ratio_threshold
        )
        early_stop = g_sum_es or g_sum_delta_es
        if isinstance(early_stop, jnp.ndarray) and early_stop.size == 1:
            # convert to specified transfer format in ic mode
            return early_stop.item()
        else:
            return early_stop

    def scale_gh(
        self, g: np.ndarray, h: np.ndarray, enable_quantization: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        if enable_quantization:
            g_scale = self.g_scale
            h_scale = self.h_scale
            return scale(g, g_scale), scale(h, h_scale)
        else:
            return g, h

    def reverse_scale_gh(
        self, g: np.ndarray, h: np.ndarray, enable_quantization: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        if enable_quantization:
            # scale 0 means abs g sum is finitely large, cannot happen.
            g_scale = self.g_scale
            h_scale = self.h_scale
            return scale(g, 1 / g_scale), scale(h, 1 / h_scale)
        else:
            return g, h


def delta_ratio(old, new):
    if old > 0:
        return abs(old - new / old)
    else:
        return 0
