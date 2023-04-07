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


import dataclasses
import os
from enum import Enum, unique
from typing import Dict, List, Union

from secretflow.device import PYU, SPU, PYUObject, SPUObject
from secretflow.utils.sigmoid import SigType


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


@dataclasses.dataclass
class PartyPath:
    party: str
    path: str


@dataclasses.dataclass
class LinearModelRecord:
    reg_type: RegType
    sig_type: SigType
    weights_spu: List[PartyPath]
    weights_pyu: List[PartyPath]


@dataclasses.dataclass
class LinearModel:
    """
    Unified linear regression model.

    Attributes:

        weights : {SPUObject, List[PYUObject]}
            for mpc lr, use SPUObject save all weights; for fl lr, use list of PYUObject.
        reg_type : RegType
            linear regression or logistic regression model.
        sig_type : SigType
            which sigmoid approximation should use, only use in mpc lr.
    """

    weights: Union[SPUObject, List[PYUObject]]
    reg_type: RegType
    sig_type: SigType

    def dump(self, dir_path: Dict[str, str]) -> LinearModelRecord:
        if isinstance(self.weights, SPUObject):
            spu_paths = [
                os.path.join(dir_path[name], 'weights')
                for name in self.weights.device.actors.keys()
            ]
            self.weights.device.dump(self.weights, spu_paths)

            weights_pyu = None
            weights_spu = [
                PartyPath(party, path)
                for party, path in zip(self.weights.device.actors.keys(), spu_paths)
            ]
        else:
            raise NotImplementedError("pyu weights are not supported")

        return LinearModelRecord(
            reg_type=self.reg_type,
            sig_type=self.sig_type,
            weights_pyu=weights_pyu,
            weights_spu=weights_spu,
        )

    @classmethod
    def load(
        cls,
        record: LinearModelRecord,
        spu: SPU = None,
        pyus: List[PYU] = None,
    ) -> 'LinearModel':
        assert len(record.weights_spu) or len(
            record.weights_pyu
        ), 'weights are not provided.'

        if record.weights_spu:
            assert spu, 'spu device is not provided'
            path_dict = {t.party: t.path for t in record.weights_spu}
            paths = [path_dict[party] for party in spu.actors.keys()]
            weights = spu.load(paths)
        else:
            raise NotImplementedError("pyu weights are not supported")

        return cls(weights, record.reg_type, record.sig_type)
