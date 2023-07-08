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

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from secretflow.device import PYUObject, reveal
from secretflow.ml.boost.sgb_v.factory.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ....core.preprocessing.params import RegType
from ....core.pure_numpy_ops.grad import (
    compute_gh_linear,
    compute_gh_logistic,
    compute_relative_scaling_factor,
    compute_sum_abs,
    scale,
)
from ..component import Component, Devices, print_params
from ..logging import LoggingParams, LoggingTools


@dataclass
class LossComputerParams:
    """
    'objective': Specify the learning objective.
        default: 'logistic'
        range: ['linear', 'logistic']
    'enable_quantization': Whether enable quantization of g and h.
        only recommended for encryption schemes with small plaintext range, like elgamal.
        default: False
        range: [True, False]
    'quantization_scale': only effective if quanization enabled. Scale the sum of g to the specified value.
        default: 10000.0
        range: [0, 10000000.0]
    'early_stop_criterion_g_abs_sum': if sum(abs(g)) is lower than or equal to this threadshold, training will stop.
        default: 0.0
        range [0.0, inf)
    'early_stop_criterion_g_abs_sum_change_ratio': if absolute g sum change ratio is lower than or equal to this threadshold, training will stop.
        default: 0.0
        range [0, 1]
    """

    objective: RegType = default_params.objective
    enable_quantization: bool = default_params.enable_quantization
    quantization_scale: float = default_params.quantization_scale
    early_stop_criterion_g_abs_sum: float = (
        default_params.early_stop_criterion_g_abs_sum
    )
    early_stop_criterion_g_abs_sum_change_ratio: float = (
        default_params.early_stop_criterion_g_abs_sum_change_ratio
    )


class LossComputer(Component):
    """Compute loss, gradients and hessians"""

    def __init__(self) -> None:
        self.params = LossComputerParams()
        self.logging_params = LoggingParams()
        self.abs_g_cache = None
        self.abs_h_cache = None
        self.last_abs_g_cache = None

    def show_params(self):
        print_params(self.params)
        print_params(self.logging_params)

    def set_params(self, params: dict):
        obj = params.get('objective', 'logistic')
        assert obj in [
            e.value for e in RegType
        ], f"objective should in {[e.value for e in RegType]}, got {obj}"
        obj = RegType(obj)

        self.params.enable_quantization = bool(
            params.get('enable_quantization', default_params.enable_quantization)
        )
        quantization_scale = float(
            params.get('quantization_scale', default_params.quantization_scale)
        )
        early_stop_criterion_g_abs_sum = float(
            params.get(
                'early_stop_criterion_g_abs_sum',
                default_params.early_stop_criterion_g_abs_sum,
            )
        )
        early_stop_criterion_g_abs_sum_change_ratio = float(
            params.get(
                'early_stop_criterion_g_abs_sum_change_ratio',
                default_params.early_stop_criterion_g_abs_sum_change_ratio,
            )
        )
        assert (
            early_stop_criterion_g_abs_sum_change_ratio >= 0
            and early_stop_criterion_g_abs_sum_change_ratio <= 1
        ), "ratio must be in range [0,1]"

        self.params.quantization_scale = quantization_scale
        self.params.early_stop_criterion_g_abs_sum = early_stop_criterion_g_abs_sum
        self.params.early_stop_criterion_g_abs_sum_change_ratio = (
            early_stop_criterion_g_abs_sum_change_ratio
        )
        self.params.objective = obj

        self.logging_params = LoggingTools.logging_params_from_dict(params)

    def get_params(self, params: dict):
        params['objective'] = self.params.objective
        params['enable_quantization'] = self.params.enable_quantization
        params['quantization_scale'] = self.params.quantization_scale
        params[
            'early_stop_criterion_g_abs_sum'
        ] = self.params.early_stop_criterion_g_abs_sum
        params[
            'early_stop_criterion_g_abs_sum_change_ratio'
        ] = self.params.early_stop_criterion_g_abs_sum_change_ratio
        LoggingTools.logging_params_write_dict(params, self.logging_params)

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, _: SGBActor):
        return

    @LoggingTools.enable_logging
    def compute_gh(
        self, y: Union[PYUObject, np.ndarray], pred: Union[PYUObject, np.ndarray]
    ) -> Tuple[PYUObject, PYUObject]:
        obj = self.params.objective
        if obj == RegType.Linear:
            g, h = self.label_holder(compute_gh_linear, num_returns=2)(y, pred)
        elif obj == RegType.Logistic:
            g, h = self.label_holder(compute_gh_logistic, num_returns=2)(y, pred)
        else:
            raise f"unknown objective {obj}"
        return g, h

    def compute_abs_sums(self, g: PYUObject, h: PYUObject):
        abs_g_sum = self.label_holder(compute_sum_abs)(g)
        abs_h_sum = self.label_holder(compute_sum_abs)(h)

        self.last_abs_g_cache = self.abs_g_cache
        self.abs_g_cache = abs_g_sum
        self.abs_h_cache = abs_h_sum

        return

    def compute_scales(self):
        scaling = self.params.quantization_scale
        abs_g_sum = self.abs_g_cache
        abs_h_sum = self.abs_h_cache
        self.g_scale = self.label_holder(compute_relative_scaling_factor)(
            abs_g_sum, scaling
        )
        self.h_scale = self.label_holder(compute_relative_scaling_factor)(
            abs_h_sum, scaling
        )
        return

    def check_abs_g_sum_early_stop(self) -> Union[bool, PYUObject]:
        threshold = self.params.early_stop_criterion_g_abs_sum
        # early stopping happened, and this is known by all parties
        abs_sum = self.abs_g_cache
        if abs_sum is None:
            return False
        return self.label_holder(lambda abs_sum, threshold: abs_sum <= threshold)(
            abs_sum, threshold
        )

    def check_abs_g_sum_change_ratio_early_stop(self) -> Union[bool, PYUObject]:
        threshold = self.params.early_stop_criterion_g_abs_sum_change_ratio
        old = self.last_abs_g_cache
        current_abs_sum = self.abs_g_cache
        if old is None or current_abs_sum is None:
            return False
        return self.label_holder(
            lambda old, abs_sum, threshold: delta_ratio(old, abs_sum) <= threshold
        )(old, current_abs_sum, threshold)

    def check_early_stop(self) -> bool:
        g_sum_es = self.check_abs_g_sum_early_stop()
        g_sum_delta_es = self.check_abs_g_sum_change_ratio_early_stop()
        return reveal(self.label_holder(lambda a, b: a or b)(g_sum_es, g_sum_delta_es))

    def scale_gh(self, g: PYUObject, h: PYUObject) -> Tuple[PYUObject, PYUObject]:
        if self.params.enable_quantization:
            g_scale = self.g_scale
            h_scale = self.h_scale
            return self.label_holder(
                lambda g, h, g_scale, h_scale: (
                    scale(g, g_scale),
                    scale(h, h_scale),
                ),
                num_returns=2,
            )(g, h, g_scale, h_scale)
        else:
            return g, h

    def reverse_scale_gh(
        self, g: PYUObject, h: PYUObject
    ) -> Tuple[PYUObject, PYUObject]:
        if self.params.enable_quantization:
            # scale 0 means abs g sum is finitely large, cannot happen.
            g_scale = self.g_scale
            h_scale = self.h_scale
            return self.label_holder(
                lambda g, h, g_scale, h_scale: (
                    scale(g, 1 / g_scale),
                    scale(h, 1 / h_scale),
                ),
                num_returns=2,
            )(g, h, g_scale, h_scale)
        else:
            return g, h


def delta_ratio(old, new):
    if old > 0:
        return abs(old - new / old)
    else:
        return 0
