# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from secretflow.device import PYU, reveal
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel


class DirectionBasedScoringAttack(AttackCallback):
    """
    Implemention of Directio-based Scoring Attack in Vertical Federated Learning: https://arxiv.org/pdf/2102.08504.
    Direction-based scoring function: Given the gradient of a positive example,
    the attacker can infer labels by calculating the cosine similarity between
    this gradient and the gradients of other samples with unknown labels.

    Args:
        attack_party: The attack party who does not have label.
        label_party: The party holds label.
        num_classes: The model classes number.
    """

    def __init__(
        self,
        attack_party: PYU,
        label_party: PYU,
        num_classes: int,
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.label_party = label_party
        self.num_classes = num_classes
        self.last_epoch = False
        self.positive_grad = None
        self.attack_logs = {}
        self.is_batch = False

    def on_epoch_begin(self, epoch=None, logs=None):
        # The attacker uses the method during the final epoch.
        if epoch == self.params['epochs'] - 1:
            self.last_epoch = True

    def on_train_batch_end(self, batch):
        # The attacker knows the label or index of a positive sample.
        # Let's assume that this sample appears in the first batch.
        if not self.last_epoch:
            return

        def record_gradient(worker: SLBaseTorchModel):
            if 'grad_lia_attack' not in worker._callback_store:
                worker._callback_store['grad_lia_attack'] = {}
            if isinstance(worker._gradient, list):
                grad = torch.stack(worker._gradient).cpu().numpy()
            else:
                grad = worker._gradient.cpu().numpy()
            if batch == 0:
                num_feature = grad.shape[1]
                worker._callback_store['grad_lia_attack']['grads'] = np.empty(
                    (0, num_feature)
                )
            worker._callback_store['grad_lia_attack']['grads'] = np.append(
                worker._callback_store['grad_lia_attack']['grads'],
                grad,
                axis=0,
            )

        def record_label(worker: SLBaseTorchModel):
            if 'grad_lia_attack' not in worker._callback_store:
                worker._callback_store['grad_lia_attack'] = {}

            if isinstance(worker.train_y, list):
                # several batch
                label = worker.train_y[0].cpu().numpy()
            else:
                # only one batch
                label = worker.train_y.cpu().numpy()

            # process the label shape to dim = 2
            if len(label.shape) == 1:
                label = label[:, np.newaxis]
            assert len(label.shape) == 2
            if self.num_classes > 2 and label.shape[1] > 1:
                # convert the one hot encoded label to the category index
                label = np.argmax(label, axis=1, keepdims=True)
            if batch == 0:
                worker._callback_store['grad_lia_attack']['labels'] = np.empty((0, 1))
            worker._callback_store['grad_lia_attack']['labels'] = np.append(
                worker._callback_store['grad_lia_attack']['labels'], label, axis=0
            )

        self._workers[self.attack_party].apply(record_gradient)
        self._workers[self.label_party].apply(record_label)

    def on_epoch_end(self, epoch=None, logs=None):
        if not self.last_epoch:
            return
        label_targets = reveal(
            self._workers[self.label_party].apply(
                lambda worker: worker._callback_store['grad_lia_attack']['labels']
            )
        )

        index = np.argmax(label_targets == 1)
        cluster_labels = self._workers[self.attack_party].apply(
            self.direction_based_scoring,
            positive_index=index,
        )
        # reveal only for compute attack acc.
        cluster_labels: List = reveal(cluster_labels)

        self.record_metrics(cluster_labels, label_targets)

    def get_attack_metrics(self):
        return self.attack_logs

    @staticmethod
    def direction_based_scoring(worker, positive_index):
        # compute cosine similarity with grad and grads.
        grad = worker._callback_store['grad_lia_attack']['grads']

        positive_grad = grad[positive_index]
        positive_grad = torch.tensor(positive_grad)
        _grad = torch.tensor(grad)

        vector_expanded = positive_grad.unsqueeze(0).expand_as(_grad)
        cosine_similarities = F.cosine_similarity(vector_expanded, _grad, dim=1)
        attack_predict = (cosine_similarities > 0).float()

        label_preds = []
        label_preds += list(attack_predict.cpu().numpy())

        return label_preds

    def record_metrics(self, label_preds: np.ndarray, label_targets: np.ndarray):
        multi_class = 'raise'
        average = 'binary'
        if self.num_classes > 2:
            if len(label_targets.shape) == 1:
                label_targets = label_targets.reshape(-1, 1)
            if len(label_preds.shape) == 1:
                label_preds = label_preds.reshape(-1, 1)
            enc = OneHotEncoder(categories=[range(self.num_classes)], sparse=False)
            label_targets = enc.fit_transform(label_targets)
            label_preds = enc.fit_transform(label_preds)
            multi_class = 'ovo'
            average = "weighted"
        if np.all(label_targets == 0) or np.all(label_targets == 1):
            self.attack_logs['attack_auc'] = -1
        else:
            self.attack_logs['attack_auc'] = roc_auc_score(
                label_targets, label_preds, multi_class=multi_class
            )
        self.attack_logs['attack_acc'] = accuracy_score(label_targets, label_preds)
        self.attack_logs['attack_recall'] = recall_score(
            label_targets, label_preds, average=average
        )
        self.attack_logs['attack_precision'] = precision_score(
            label_targets, label_preds, average=average
        )
