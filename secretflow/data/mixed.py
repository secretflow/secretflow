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

from dataclasses import dataclass
from typing import List, Union

from .base import DataFrameBase, NdArrayBase
from .horizontal import HDataFrame, HNdArray
from .vertical import VDataFrame, VNdArray


@dataclass
class MixDataFrame(DataFrameBase):
    """混合DataFrame"""
    partitions: List[Union[HDataFrame, VDataFrame]]


@dataclass
class MixNdArray(NdArrayBase):
    """混合NdArray"""
    partitions: List[Union[HNdArray, VNdArray]]
