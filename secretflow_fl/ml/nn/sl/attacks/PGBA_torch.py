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


import os
import sys
from typing import Callable, Dict, List

import numpy as np
import torch
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.core.torch import BuilderType
from secretflow_fl.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from torch.utils.data import DataLoader

from secretflow import reveal
from secretflow.device import PYU


def angular_distance(logits1, logits2):
    """Calculate cosine similarity between two vectors"""
    numerator = logits1.mul(logits2).sum()
    logits1_l2norm = logits1.mul(logits1).sum().sqrt()
    logits2_l2norm = logits2.mul(logits2).sum().sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    return torch.div(numerator, denominator)


class PGBAAttackCallback(AttackCallback):
    """
    Implementation of Practical and General Backdoor Attacks Against Vertical Federated Learning: https://arxiv.org/abs/2306.10746

    Attributes:
        attack_party: The party that launches the attack
        victim_party: The party that is targeted by the attack
        target_idx: Target sample index
        num: Number of samples to change in each batch
        thre: Threshold of similarity
        poison_num: Max number of total poison sample
        trigger_words: Trigger pattern to insert
        base_model_list: List of party, which decide input order of models
        data_builder: Function to build data loaders
        vocab: Dictionary mapping words to token IDs
        batch_size: Batch size during training
        seed: Random seed for reproducibility
        device: Device for computation (CPU/GPU)
    """

    def __init__(
        self,
        attack_party: PYU,
        victim_party: PYU,
        target_idx: int,
        num: int,
        thre: float,
        poison_num: int,
        trigger_words: str,
        base_model_list: List[PYU],
        data_builder: Callable,
        vocab: Dict,
        batch_size: int = 64,
        seed: int = 1,
        device: str = "cpu",
        **params
    ):
        super().__init__(**params)
        self.attack_party = attack_party
        self.victim_party = victim_party
        self.target_idx = target_idx
        self.num = num
        self.thre = thre
        self.poison_num = poison_num
        self.trigger_words = trigger_words
        self.base_model_list = [p.party for p in base_model_list]
        self.data_builder = data_builder
        self.vocab = vocab
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.metrics = None
        self.cluster = {}
        self.other_class = None
        self.target_label = None
        self.count = 0
        self.target_grad = None
        self.if_SDP = False
        self.batch = 0
        self.epoch = 0
        self.target_data = None

    def on_train_begin(self, logs=None):
        """Initialize backdoor attacker on the attack party"""
        print("it's on_train_begin now!")

        def init_attacker(
            attack_worker: SLBaseTorchModel,
            backdoor_attacker: PGBAAttacker,
        ):
            def get_model_fuse(attack_worker: SLBaseTorchModel):
                return attack_worker.model_fuse

            attack_worker.attacker = backdoor_attacker
            attack_worker.attacker.model_base = attack_worker.model_base
            model_fuse = reveal(self._workers[self.victim_party].apply(get_model_fuse))
            attack_worker.attacker.model_top = model_fuse

        attacker = PGBAAttacker(
            target_idx=self.target_idx,
            num=self.num,
            thre=self.thre,
            poison_num=self.poison_num,
            trigger_words=self.trigger_words,
            data_builder=self.data_builder,
            vocab=self.vocab,
            batch_size=self.batch_size,
            seed=self.seed,
            device=self.device,
        )

        self._workers[self.attack_party].apply(init_attacker, attacker)

    def on_epoch_begin(self, epoch=None, logs=None):
        """Initialize epoch"""
        self.epoch = epoch

    def on_train_batch_begin(self, batch=None):
        """Initialize batch"""
        self.batch = batch

    def on_base_forward_begin(self):
        """Perform Feature Replacement and Source Data Perturbation"""
        status = self._workers[self.attack_party].get_traing_status()

        def inject_backdoor(attack_worker, batch_idx, cluster):
            if batch_idx in cluster:
                return attack_worker.attacker.inject_backdoor(
                    batch_idx, cluster[batch_idx]
                )
            return None

        batch = self.batch
        epoch = self.epoch

        # When epoch = 1, perform feature replacements.
        if epoch == 1:

            def modify_batch_data(attack_worker, batch_idx):
                current_data = attack_worker._data_x
                if current_data.size(0) < self.batch_size:
                    return True
                result = attack_worker.attacker.feature_replacements(
                    current_data, self.target_data, self.batch
                )
                attack_worker._data_x = result
                attack_worker.train_x = [result]
                return True

            result = self._workers[self.attack_party].apply(
                modify_batch_data, self.batch
            )
            a = 1

        # When epoch > 3, insert backdoor samples.
        if epoch > 3:
            if self.batch in self.cluster and status.data["stage"] == "train":

                def modify_batch_data(attack_worker, batch_idx, cluster_indices):
                    current_data = attack_worker._data_x
                    if current_data.size(0) < self.batch_size:
                        return True
                    result = self._workers[self.attack_party].apply(
                        inject_backdoor, self.batch, self.cluster
                    )
                    modified_data, modified_target = reveal(result)
                    modified_data_list = [modified_data]
                    attack_worker._data_x = modified_data
                    attack_worker.train_x = modified_data_list
                    return True

                self._workers[self.attack_party].apply(
                    modify_batch_data, self.batch, self.cluster[self.batch]
                )

    def on_base_backward_end(self):
        """Perform Source Data Detection (SDD)"""
        print("it's on_base_backward_end now!")
        status = self._workers[self.attack_party].get_traing_status()

        def get_target(attack_worker):
            current_gradients = attack_worker._gradient
            self.target_grad = current_gradients[self.target_idx]
            self.target_data = attack_worker._data_x[self.target_idx]

        def get_cluster(attack_worker):
            if attack_worker._data_x.size(0) < self.batch_size:
                return True
            if self.batch == 0:
                self.target_grad = attack_worker._gradient[self.target_idx]
            self.cluster = attack_worker.attacker.get_cluster(
                self.target_grad, attack_worker._gradient, self.batch
            )
            return True

        if self.batch == 0 and status.data["stage"] == "train" and self.epoch == 0:
            self._workers[self.attack_party].apply(get_target)

        # Similarity Computation
        if 2 <= self.epoch <= 3 and status.data["stage"] == "train":
            self._workers[self.attack_party].apply(get_cluster)

    def get_attack_metrics(self):
        return self.metrics


class PGBAAttacker:
    """
    Implementation of backdoor attacker for IMDB sentiment analysis in VFL
    Attributes:
        target_idx: Target sample index
        num: Number of samples to change in each batch
        thre: Threshold of similarity
        poison_num: Max number of total poison sample
        trigger_words: Trigger pattern to insert
        data_builder: Function to build data loaders
        vocab: Dictionary mapping words to token IDs
        batch_size: Batch size during training
        seed: Random seed for reproducibility
        device: Device for computation (CPU/GPU)
    """

    def __init__(
        self,
        target_idx: int,
        num: int,
        thre: float,
        poison_num: int,
        trigger_words: str,
        data_builder: Callable,
        vocab: Dict,
        batch_size: int = 64,
        seed: int = 1,
        device: str = "cpu",
    ):
        self.target_idx = target_idx
        self.num = num
        self.thre = thre
        self.poison_num = poison_num
        self.trigger_words = trigger_words
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.vocab = vocab

        # Load data
        train_dataset_right, test_dataset_right = data_builder()

        self.train_dataset_right = train_dataset_right
        self.test_dataset_right = test_dataset_right

        # Create data loaders
        self.train_loader_right = DataLoader(
            train_dataset_right, batch_size=batch_size, shuffle=False, drop_last=True
        )
        self.test_loader_right = DataLoader(
            test_dataset_right, batch_size=batch_size, shuffle=False, drop_last=True
        )

        # For tracking current batch
        self.current_batch_idx = 0
        self.target_grad = None
        self.cluster = {}
        self.count = 0
        self.if_changed = None
        self.other_class = None

        max_samples = len(self.train_dataset_right)
        self.if_changed = np.zeros(max_samples)
        self.other_class = np.zeros(max_samples)
        self.change_idx = None

    def set_base_model(
        self, base_model: torch.nn.Module, builder_base: BuilderType = None
    ):
        self.base_model = base_model
        self.builder_base = builder_base

    def infect_text(self, text, trigger_words):
        """Add trigger pattern to text data"""
        tokenized = text.clone()

        trigger_tokens = []
        for word in trigger_words.split():
            if word in self.vocab:
                trigger_tokens.append(self.vocab[word])
            else:
                trigger_tokens.append(self.vocab["<unk>"])

        insertion_point = 5

        new_text = torch.cat(
            [
                tokenized[:insertion_point],
                torch.tensor(trigger_tokens, device=tokenized.device),
                tokenized[insertion_point : len(tokenized) - len(trigger_tokens)],
            ]
        )

        return new_text

    def feature_replacements(self, data, target_data, batch_idx):
        """feature_replacements in Source Data Detection (SDD)"""
        if self.count < self.poison_num:
            # Select samples to replace with target sample
            choice_idx = np.where(
                self.if_changed[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                == 0
            )[0]
            if len(choice_idx) >= self.num:
                self.change_idx = np.random.choice(choice_idx, self.num, replace=False)
                # Replace features with target features
                data[self.change_idx] = target_data
                self.if_changed[batch_idx * self.batch_size + self.change_idx] = 1

        return data

    def get_cluster(self, target_grad, batch_grad, batch_idx):
        """Source Data Detection (SDD)"""
        # Find samples with similar gradients to target
        if self.count < self.poison_num:
            same_class_idx = []
            for i in self.change_idx:
                simi = angular_distance(target_grad, batch_grad[i])
                if simi > self.thre:
                    same_class_idx.append(i)
                    self.other_class[batch_idx * self.batch_size + i] = 1

            if len(same_class_idx) != 0:
                self.count = self.count + len(same_class_idx)
                if self.count > self.poison_num:
                    excess_num = self.count - self.poison_num
                    del same_class_idx[-excess_num:]
                    self.count = self.count - excess_num

                if batch_idx in self.cluster.keys():
                    self.cluster[batch_idx] = self.cluster[batch_idx] + same_class_idx
                else:
                    self.cluster[batch_idx] = same_class_idx

        return self.cluster

    def inject_backdoor(self, batch_idx, indices):
        """
        Source Data Perturbation (SDP) phase
        """
        train_iter_right = iter(self.train_loader_right)
        for i in range(batch_idx + 1):
            data2, target = next(train_iter_right)

        data2, target = data2.to(self.device), target.to(self.device)

        indices = [int(i) if not isinstance(i, int) else i for i in indices]

        other_class_idx = np.where(self.other_class == 0)[0]

        for i in indices:
            j = np.random.choice(other_class_idx, replace=False)
            target_text, _ = self.train_dataset_right[j]
            target_text = target_text.to(self.device)

            data2[i] = self.infect_text(
                target_text.clone().detach(), self.trigger_words
            )

        return data2, target
