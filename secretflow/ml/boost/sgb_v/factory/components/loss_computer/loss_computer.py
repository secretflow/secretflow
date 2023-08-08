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
from typing import Tuple, Union, List

import numpy as np

from secretflow.device import PYUObject
from secretflow.ml.boost.sgb_v.core.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ....core.params import RegType
from ..component import Component, Devices, print_params
from ..logging import LoggingParams, LoggingTools
from .loss_computer_actor import LossComputerActor


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
        obj = RegType(obj)

        self.params.enable_quantization = params.get(
            'enable_quantization', default_params.enable_quantization
        )
        quantization_scale = params.get(
            'quantization_scale', default_params.quantization_scale
        )

        early_stop_criterion_g_abs_sum = params.get(
            'early_stop_criterion_g_abs_sum',
            default_params.early_stop_criterion_g_abs_sum,
        )

        early_stop_criterion_g_abs_sum_change_ratio = params.get(
            'early_stop_criterion_g_abs_sum_change_ratio',
            default_params.early_stop_criterion_g_abs_sum_change_ratio,
        )

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

    def set_actors(self, actors: List[SGBActor]):
        for actor in actors:
            if actor.device == self.label_holder:
                self.actor = actor
                break
        self.actor.register_class('LossComputerActor', LossComputerActor)
        return

    def del_actors(self):
        del self.actor
        return

    @LoggingTools.enable_logging
    def compute_gh(
        self, y: Union[PYUObject, np.ndarray], pred: Union[PYUObject, np.ndarray]
    ) -> Tuple[PYUObject, PYUObject]:
        obj = self.params.objective
        return self.actor.invoke_class_method_two_ret(
            'LossComputerActor', 'compute_gh', y, pred, obj
        )

    def compute_abs_sums(self, g: PYUObject, h: PYUObject):
        return self.actor.invoke_class_method(
            'LossComputerActor', 'compute_abs_sums', g, h
        )

    def compute_scales(self):
        scaling = self.params.quantization_scale
        return self.actor.invoke_class_method(
            'LossComputerActor', 'compute_scales', scaling
        )

    def check_early_stop(self) -> PYUObject:
        abs_sum_threshold = self.params.early_stop_criterion_g_abs_sum
        abs_sum_change_ratio_threshold = (
            self.params.early_stop_criterion_g_abs_sum_change_ratio
        )
        return self.actor.invoke_class_method(
            'LossComputerActor',
            'check_early_stop',
            abs_sum_threshold,
            abs_sum_change_ratio_threshold,
        )

    def scale_gh(self, g: PYUObject, h: PYUObject) -> Tuple[PYUObject, PYUObject]:
        enable_quantization = self.params.enable_quantization
        return self.actor.invoke_class_method_two_ret(
            'LossComputerActor', 'scale_gh', g, h, enable_quantization
        )

    def reverse_scale_gh(
        self, g: PYUObject, h: PYUObject
    ) -> Tuple[PYUObject, PYUObject]:
        enable_quantization = self.params.enable_quantization
        return self.actor.invoke_class_method_two_ret(
            'LossComputerActor', 'reverse_scale_gh', g, h, enable_quantization
        )
