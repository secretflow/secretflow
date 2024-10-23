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

import logging
from typing import List

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from secretflow.device import PYU, reveal
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel

_MAX_GROUP_SIZE = 30000


def get_suitable_steps(length: int) -> List[int]:
    """
    To prevent memory overflow, we need to split data into several parts with suitable size.
    Args:
        length: Total length of the data.

    Returns:
        List[int]: List of steps where each group has a suitable size closed to _MAX_GROUP_SIZE.
    """
    group_size = _MAX_GROUP_SIZE
    groups = length // _MAX_GROUP_SIZE
    if length % _MAX_GROUP_SIZE != 0:
        groups += 1
        group_size = length // groups
    steps = [
        group_size + 1 if i < length % group_size else group_size for i in range(groups)
    ]
    assert (
        sum(steps) == length
    ), f"group sums and total length not match, sum = {sum(steps)}, length = {length}"
    return steps


class GradientClusterLabelInferenceAttack(AttackCallback):
    """
    label inference attack under semi honesty model.

    The inspiration for this attack comes from a fact in neural networks that is easy to overlook.
    As the input samples propagate through the neural network,
    the intermediate results of samples with the same label become increasingly similar,
    ultimately producing identical labels.
    This is probably because, during the forward propagation process,
    the model continuously discards information irrelevant to classification and reinforces information relevanted.
    Since the information associated with the same label is likely to be the same,
    their intermediate results are similar as well.
    This method is to infer the label based on the gradient direction of the intermediate results.

    Args:
        attack_party: The attack party who does not have label.
        label_party: The party holds label.
        num_classes: The model classes number.
        n_neighbors: Number of neighbors when constructing the affinity matrix using the nearest neighbors method.
        n_jobs: Parallelize the computation.
    """

    def __init__(
        self,
        attack_party: PYU,
        label_party: PYU,
        num_classes: int,
        n_neighbors: int = 4,
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.label_party = label_party
        self.num_classes = num_classes
        self.n_neighbors = n_neighbors
        self.n_jobs = -1
        self.last_epoch = False
        self.attack_logs = {}

    def on_epoch_begin(self, epoch=None, logs=None):
        if epoch == self.params['epochs'] - 1:
            self.last_epoch = True

        def append_callback_store(worker):
            worker._callback_store['grad_lia_attack'] = {}
            worker._callback_store['grad_lia_attack']['labels'] = np.empty((0, 1))

        self._workers[self.label_party].apply(append_callback_store)

    def on_base_forward_begin(self):
        # record label on base forward begin, to prevent the train_y being modified.
        def record_label(worker: SLBaseTorchModel):
            if worker.model_base.training:
                # only processs in train mode.
                label = worker.train_y.cpu().detach().numpy()
                # process the label shape to dim = 2ss
                if len(label.shape) == 1:
                    label = label[:, np.newaxis]
                assert len(label.shape) == 2
                if label.shape[1] > 1:
                    # convert the one hot encoded label to the category index
                    label = np.argmax(label, axis=1, keepdims=True)
                worker._callback_store['grad_lia_attack']['labels'] = np.append(
                    worker._callback_store['grad_lia_attack']['labels'], label, axis=0
                )

        self._workers[self.label_party].apply(record_label)

    def on_train_batch_end(self, batch):
        if not self.last_epoch:
            return

        def record_gradient(worker: SLBaseTorchModel):
            if 'grad_lia_attack' not in worker._callback_store:
                worker._callback_store['grad_lia_attack'] = {}
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

        self._workers[self.attack_party].apply(record_gradient)

    def on_epoch_end(self, epoch=None, logs=None):
        if not self.last_epoch:
            return
        # infert label by clustering algorithm.
        cluster_labels = self._workers[self.attack_party].apply(
            self.spectral_cluster_label,
            num_classes=self.num_classes,
            n_neighbors=self.n_neighbors,
            affinity='precomputed',
            n_jobs=self.n_jobs,
        )
        # reveal only for compute attack acc.
        cluster_labels: List = reveal(cluster_labels)
        label_targets = reveal(
            self._workers[self.label_party].apply(
                lambda worker: worker._callback_store['grad_lia_attack']['labels']
            )
        )
        label_preds = self.map_predict_labels(cluster_labels, label_targets)
        self.record_metrics(label_preds, label_targets)

    def get_attack_metrics(self):
        return self.attack_logs

    def map_predict_labels(
        self, cluster_labels: List[np.ndarray], label_targets: np.ndarray
    ):
        """
        Map all cluster predict labels into real labels.
        Args:
            cluster_labels: The clustering results of a list which are independent of each other.
            label_targets: The real labels.

        Returns:
            The mapped labels with the cluster labels.
        """
        label_preds = None
        offset = 0
        for pred_labels in cluster_labels:
            affine_list = []
            label_target = label_targets[offset : offset + pred_labels.shape[0]]
            offset += pred_labels.shape[0]
            for i in range(self.num_classes):
                preds = np.argmax(
                    np.bincount(list(map(int, label_target[pred_labels == i])))
                )
                affine_list.append(preds)
            preds = np.array([affine_list[x] for x in pred_labels])
            label_preds = (
                preds if label_preds is None else np.append(label_preds, preds, axis=0)
            )
            if 'affine_list' not in self.attack_logs:
                self.attack_logs['affine_list'] = affine_list
        return label_preds

    @staticmethod
    def spectral_cluster_label(worker, num_classes, n_neighbors, affinity, n_jobs):
        # compute cos similarity with grad and grads.
        grad = worker._callback_store['grad_lia_attack']['grads']
        steps = get_suitable_steps(grad.shape[0])
        start = 0
        label_preds = []
        for i, step in enumerate(steps):
            logging.info(
                f"handle the spectral cluster label with start = {start}, step = {step}, rounds {i}/{len(steps)}"
            )
            work_grad = grad[start : start + step]
            start += step
            sim_matrix = 1 + cosine_similarity(work_grad, work_grad)
            preds = SpectralClustering(
                n_clusters=num_classes,
                n_neighbors=n_neighbors,
                affinity=affinity,
                n_jobs=n_jobs,
            ).fit_predict(sim_matrix)
            label_preds.append(preds)
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
        if np.all(label_targets == label_targets[0]):
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
