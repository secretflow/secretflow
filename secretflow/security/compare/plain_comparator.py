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

from secretflow.device import PYU
from secretflow.security.compare.device_comparator import DeviceComparator


class PlainComparator(DeviceComparator):
    """Plaintext compartator.

    The computation will be performed in plaintext.

    Warnings:
        PlainAggregator is for debugging purpose only.
        You should not use it in production.

    Examples:
        >>> import numpy as np
        # Alice and bob are both pyu instances.
        >>> a = alice(lambda : np.random.rand(2, 5))()
        >>> b = bob(lambda : np.random.rand(2, 5))()
        >>> comparator = PlainComparator(alice)
        >>> min_a_b = comparator.min([a, b], axis=0)
        >>> sf.reveal(min_a_b)
        array([[0.47092903, 0.77865475, 0.05917433, 0.07155096, 0.16089967],
            [0.56598   , 0.51047045, 0.35771865, 0.23004009, 0.23400909]],
            dtype=float32)
            >>> max_a_b = comparator.max([a, b], axis=0)
        >>> sf.reveal(max_a_b)
        array([[0.5939065 , 0.8463326 , 0.14722177, 0.9977698 , 0.6186677 ],
            [0.65607053, 0.611439  , 0.957074  , 0.6548823 , 0.445968  ]],
            dtype=float32)

    """

    def __post_init__(self):
        assert isinstance(
            self.device, PYU
        ), f'Accepts PYU only but got {type(self.device)}.'
