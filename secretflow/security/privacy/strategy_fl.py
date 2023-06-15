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

from secretflow.security.privacy.mechanism.mechanism_fl import GaussianModelDP


class DPStrategyFL:
    def __init__(
        self,
        model_gdp: GaussianModelDP = None,
        accountant_type='rdp',
    ):
        """
        Args:
            model_gdp: global dp strategy on model parameters or gradients.
            accountant_type: Method of calculating accountant, only supports "rdp".
        """
        self.model_gdp = model_gdp

        if accountant_type == 'rdp':
            self.accountant_type = accountant_type

    def get_privacy_spent(self, step: int, orders=None):
        """Get accountant of all used dp mechanism.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """

        privacy_dict = {}
        if self.model_gdp is not None:
            if self.accountant_type == 'rdp':
                model_eps, model_delta, _ = self.model_gdp.privacy_spent_rdp(
                    step, orders
                )
            else:
                raise ValueError('the accountant_type only supports "rdp".')

            privacy_dict['model_eps'] = model_eps
            privacy_dict['model_delta'] = model_delta

        return privacy_dict
