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

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
from heu import phe

from secretflow.device import PYU, HEUObject, PYUObject
from secretflow.device.device.heu import HEUMoveConfig

from ..component import Component, Devices, print_params


@dataclass
class GradientEncryptorParams:
    """
    'fixed_point_parameter': int. Any floating point number encoded by heu,
             will multiply a scale and take the round,
             scale = 2 ** fixed_point_parameter.
             larger value may mean more numerical accurate,
             but too large will lead to overflow problem.
             See HEU's document for more details.
        default: 20

    'batch_encoding_enabled': bool. if use batch encoding optimization.
        default: True.

    'audit_paths': dict. {device : path to save log for audit}
    """

    fixed_point_parameter: int = 20
    batch_encoding_enabled: bool = True
    audit_paths: dict = field(default_factory=dict)


def define_encoder(params: GradientEncryptorParams):
    fxp_scale = np.power(2, params.fixed_point_parameter)
    if params.batch_encoding_enabled:
        return phe.BatchFloatEncoderParams(scale=fxp_scale)
    else:
        return phe.FloatEncoderParams(scale=fxp_scale)


class GradientEncryptor(Component):
    """Manage all encryptions related to y, gradients, hessians"""

    def __init__(self):
        self.params = GradientEncryptorParams()
        self.gh_encoder = define_encoder(self.params)

    def show_params(self):
        print_params(self.params)

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder
        self.workers = devices.workers
        self.heu = devices.heu
        label_holder_party_name = self.label_holder.party
        assert (
            label_holder_party_name == self.heu.sk_keeper_name()
        ), f"HEU sk keeper party {self.heu.sk_keeper_name()}, mismatch with label_holder device's party {label_holder_party_name}"

    def get_params(self, params: dict):
        params['fixed_point_parameter'] = self.params.fixed_point_parameter
        params['batch_encoding_enabled'] = self.params.batch_encoding_enabled
        params['audit_paths'] = self.params.audit_paths

    def set_params(self, params: dict):
        # validation
        fxp_r = params.get('fixed_point_parameter', 20)
        assert (
            fxp_r >= 20 and fxp_r <= 100
        ), f"fixed_point_parameter should in [20, 100], got {fxp_r}"

        enable_batch_encoding = bool(params.get('batch_encoding_enabled', True))

        audit_paths = params.get('audit_paths', {})
        assert isinstance(audit_paths, dict), " audit paths must be a dict"

        # set params
        self.params.fixed_point_parameter = fxp_r
        self.params.batch_encoding_enabled = enable_batch_encoding
        self.params.audit_paths = audit_paths

        # calculate attributes
        self.gh_encoder = define_encoder(self.params)

    def pack(self, g: PYUObject, h: PYUObject) -> PYUObject:
        return self.label_holder(lambda g, h: np.concatenate([g, h], axis=1))(g, h)

    def encrypt(self, gh: PYUObject, tree_index: int) -> HEUObject:
        if self.label_holder.party in self.params.audit_paths:
            path = (
                self.params.audit_paths[self.label_holder.party]
                + ".tree_"
                + str(tree_index)
            )
        else:
            path = None

        return gh.to(self.heu, move_config(self.label_holder, self.gh_encoder)).encrypt(
            path
        )

    def cache_to_workers(
        self, encrypted_gh: HEUObject, gh: PYUObject
    ) -> Dict[PYU, Union[HEUObject, PYUObject]]:
        cache = {
            worker: encrypted_gh.to(self.heu, move_config(worker, self.gh_encoder))
            for worker in self.workers
            if worker != self.label_holder
        }
        cache[self.label_holder] = gh
        return cache

    def get_move_config(self, pyu):
        return move_config(pyu, self.gh_encoder)


def move_config(pyu, params):
    move_config = HEUMoveConfig()
    if isinstance(pyu, PYU):
        move_config.heu_encoder = params
        move_config.heu_dest_party = pyu.party
    return move_config
