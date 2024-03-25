# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, List, Union


@dataclass
class ForwardData:
    """
    ForwardData is a dataclass for data uploaded by each party to label party for computation.

    hidden: base model hidden layers outputs
    losses: the sum of base model losses should added up to fuse model loss
    """

    hidden: Union[Any, List[Any]] = None
    losses: Any = None
