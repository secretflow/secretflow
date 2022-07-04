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

from secretflow.security.privacy.mechanism.tensorflow.layers import EmbeddingDP, LabelDP


class DPStrategy:
    def __init__(self, embedding_dp: EmbeddingDP = None, label_dp: LabelDP = None):
        """
        Args:
            embedding_dp: Embedding dp layer.
            label_dp: Label dp layer.
        """
        self.embedding_dp = embedding_dp
        self.label_dp = label_dp

    def get_privacy_spent(self, step: int, orders=None):
        """Get accountant of all used dp mechanism.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """

        privacy_dict = {}
        if self.embedding_dp is not None:
            emb_eps, emb_delta, _ = self.embedding_dp.privacy_spent_rdp(step, orders)
            privacy_dict['emb_eps'] = emb_eps
            privacy_dict['emb_delta'] = emb_delta

        if self.label_dp is not None:
            privacy_dict['label_eps'] = self.label_dp.privacy_spent()
        return privacy_dict
