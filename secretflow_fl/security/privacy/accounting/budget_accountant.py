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

from abc import ABC
from typing import List, Optional

from secretflow_fl.security.privacy.accounting.gdp_accountant import (
    cal_mu_poisson,
    cal_mu_uniform,
    get_eps_from_mu,
)
from secretflow_fl.security.privacy.accounting.rdp_accountant import (
    DEFAULT_RDP_ORDERS,
    get_privacy_spent_rdp,
    get_rdp,
)


class BudgetAccountant(ABC):
    batch_size: int
    num_samples: int
    noise_multiplier: float
    delta: float

    def __init__(self) -> None:
        super().__init__()

    def privacy_spent_rdp(self, step: int, orders: Optional[List[float]] = None):
        """Get accountant using RDP.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """

        if orders is None:
            orders = DEFAULT_RDP_ORDERS

        q = self.batch_size / self.num_samples
        rdp = get_rdp(q, self.noise_multiplier, step, orders)
        eps, _, opt_order = get_privacy_spent_rdp(orders, rdp, target_delta=self.delta)
        return eps, self.delta, opt_order

    def privacy_spent_gdp(
        self,
        step: int,
        sampling_type: str,
    ):
        """Get accountant using GDP.

        Args:
            step: The current step of model training or prediction.
            sampling_type: Sampling type, which must be "poisson" or "uniform".
        """

        if sampling_type == 'poisson':
            mu_ideal = cal_mu_poisson(
                step, self.noise_multiplier, self.num_samples, self.batch_size
            )
        elif sampling_type == 'uniform':
            mu_ideal = cal_mu_uniform(
                step, self.noise_multiplier, self.num_samples, self.batch_size
            )
        else:
            raise ValueError('the sampling_type must be "poisson" or "uniform".')

        eps = get_eps_from_mu(mu_ideal, self.delta)
        return eps, self.delta
