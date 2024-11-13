# Copyright 2023 Ant Group Co., Ltd.
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

import numpy as np


class LabelDP:
    """Label differential privacy perturbation"""

    def __init__(self, eps: float) -> None:
        """
        Args:
            eps: epsilon for pure DP.
        """
        self._eps = eps

    def __call__(self, inputs: np.ndarray):
        """Random Response. Except for binary classification, inputs only support onehot form.

        Args:
            inputs: the label.
        """

        def _case_binary(inputs):
            p_ori = np.exp(self._eps) / (np.exp(self._eps) + 1)
            choice_ori = np.random.binomial(1, p_ori, size=inputs.shape[0])
            outputs = np.abs(1 - choice_ori - inputs)
            return outputs

        def _case_onehot(inputs):
            p_ori = np.exp(self._eps) / (np.exp(self._eps) + inputs.shape[-1] - 1)
            p_oth = (1 - p_ori) / (inputs.shape[-1] - 1)
            p_array = inputs * (p_ori - p_oth) + np.ones(inputs.shape) * p_oth
            index_rr = np.array(
                [
                    np.random.choice(inputs.shape[-1], p=p_array[i])
                    for i in range(inputs.shape[0])
                ]
            )
            outputs = np.eye(inputs.shape[-1])[index_rr]
            return outputs

        if not np.sum((inputs == 0) + (inputs == 1)) == inputs.size:
            raise ValueError(
                'Except for binary classification, inputs only support onehot form.'
            )

        if inputs.ndim == 1:
            outputs = _case_binary(inputs=inputs)
        elif inputs.ndim == 2:
            if inputs.shape[-1] == 1:
                outputs = _case_binary(inputs=inputs.reshape(-1))
                outputs = outputs.reshape(-1, 1)
            else:
                outputs = _case_onehot(inputs=inputs)
        else:
            raise ValueError('the dim of inputs in LabelDP must be less than 2.')

        # TODO(@yushi): Support regression.
        return outputs

    def privacy_spent(self):
        return self._eps
