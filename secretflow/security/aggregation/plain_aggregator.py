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
from secretflow.security.aggregation.device_aggregator import DeviceAggregator


class PlainAggregator(DeviceAggregator):
    """Plaintext aggregator.

    The computation will be performed in plaintext.

    Warnings:
        PlainAggregator is for debugging purpose only.
        You should not use it in production.

    Examples:
      >>> # Alice and bob are both pyu instances.
      >>> aggregator = PlainAggregator(alice)
      >>> a = alice(lambda : np.random.rand(2, 5))()
      >>> b = bob(lambda : np.random.rand(2, 5))()
      >>> sum_a_b = aggregator.sum([a, b], axis=0)
      >>> # Get the result.
      >>> sf.reveal(sum_a_b)
      array([[0.5954927 , 0.9381409 , 0.99397117, 1.551537  , 0.32698634],
        [1.288345  , 1.1820003 , 1.1769378 , 0.7396539 , 1.215364  ]],
        dtype=float32)
      >>> average_a_b = aggregator.average([a, b], axis=0)
      >>> sf.reveal(average_a_b)
      array([[0.29774636, 0.46907046, 0.49698558, 0.7757685 , 0.16349317],
        [0.6441725 , 0.59100014, 0.5884689 , 0.36982694, 0.607682  ]],
        dtype=float32)

    """

    def __post_init__(self):
        assert isinstance(
            self.device, PYU
        ), f'Accepts PYU only but got {type(self.device)}.'
