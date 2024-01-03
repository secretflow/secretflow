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

from typing import List

import numpy as np
import torch

from secretflow.device import PYU
from secretflow.ml.nn.callbacks.attack import AttackCallback


class ReplayAttack(AttackCallback):
    """
    Implemention of backdoor attack algorithm: Replay Attack.
    In Replay Attack, we want to make poison samples' prediction be target label. We assume that same target samples(samples with target label) has already been collected. During training, we gather the hidden outputs of the target samples. In the prediction process, we use the collected hidden outputs of the target samples to replace the hidden outputs of the poison samples, causing the prediction of the poison samples to become the target label.
    Attributes:
        attack_party: attack party.
        target_idx: list of target samples' indexes, dataloader should return (data, index).
        poison_idx: list of poison samples' indexes.
        hidden_size: hidden output's shape, without batch_size.
    """

    def __init__(
        self,
        attack_party: PYU,
        target_idx: List[int],
        poison_idx: List[int],
        hidden_size: List[int],
        batch_size: int,
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.target_idx = target_idx
        self.poison_idx = poison_idx
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.target_offsets = None
        self.poison_offsets = None

    def on_train_begin(self, logs):
        def init_callback_store(attack_worker, target_len, hidden_size):
            attack_worker._callback_store['replay_attack'] = {}
            attack_worker._callback_store['replay_attack'][
                'train_target_hiddens'
            ] = np.zeros([target_len] + hidden_size)
            attack_worker._callback_store['replay_attack']['record_counter'] = 0

        self._workers[self.attack_party].apply(
            init_callback_store, len(self.target_idx), self.hidden_size
        )

    def on_train_batch_begin(self, batch):
        data_idx = [
            x for x in range(batch * self.batch_size, (batch + 1) * self.batch_size)
        ]

        target_set = np.intersect1d(data_idx, self.target_idx)
        self.target_offsets = np.where(np.isin(data_idx, target_set))[0]

        poison_set = np.intersect1d(data_idx, self.poison_idx)
        self.poison_offsets = np.where(np.isin(data_idx, poison_set))[0]

    def on_after_base_forward(self):
        def record_and_replay(
            attack_worker, target_len, target_offsets, poison_offsets
        ):
            att_info = attack_worker._callback_store['replay_attack']
            if attack_worker._training:
                tlen = len(target_offsets)
                # record target embeddings
                if tlen > 0:
                    hidden_np = attack_worker._h.detach().numpy()
                    batch_hiddens = hidden_np[target_offsets]
                    cnt = att_info['record_counter']
                    att_info['train_target_hiddens'][cnt : cnt + tlen] = batch_hiddens

                    cnt += tlen
                    if cnt >= target_len:
                        cnt -= target_len
                    att_info['record_counter'] = cnt
            else:
                plen = len(poison_offsets)
                # replay: target embeddings -> poison embeddings
                if plen > 0:
                    hiddens_np = attack_worker._h.numpy()
                    replay_keys = np.random.choice(
                        np.arange(target_len), (plen,), replace=True
                    )
                    hiddens_np[poison_offsets] = att_info['train_target_hiddens'][
                        replay_keys
                    ]
                    attack_worker._h = torch.tensor(hiddens_np)

        if len(self.target_offsets) > 0 or len(self.poison_offsets) > 0:
            self._workers[self.attack_party].apply(
                record_and_replay,
                len(self.target_idx),
                self.target_offsets,
                self.poison_offsets,
            )
