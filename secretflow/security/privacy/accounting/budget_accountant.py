from abc import ABC
from typing import List

from secretflow.security.privacy.accounting.gdp_accountant import (
    cal_mu_poisson,
    cal_mu_uniform,
    get_eps_from_mu,
)
from secretflow.security.privacy.accounting.rdp_accountant import (
    get_privacy_spent_rdp,
    get_rdp,
)


class BudgetAccountant(ABC):
    def __init__(self) -> None:
        super().__init__()

    def privacy_spent_rdp(self, step: int, orders: List = None):
        """Get accountant using RDP.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """

        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

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
