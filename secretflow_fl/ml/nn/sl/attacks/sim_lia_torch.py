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

"""
This file references code of paper : Similarity-based Label Inference Attack against Training and Inference of Split Learning(IEEE2024)
https://ieeexplore.ieee.org/document/10411061
"""

import numpy as np

from secretflow.device import PYU, reveal, wait
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import sklearn.preprocessing as preprocessing


def cosine_distance(x: np.array, y: np.array):
    # cosine distance. Use -1 to convert cosine similarity to distance
    return -np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def euclidean_distance(x: np.array, y: np.array):
    # euclidean distance
    return np.linalg.norm(x - y)


def distance_based(measure="cosine"):
    def distance_attack(
        collect_data: np.array, known_data: np.array, known_label: dict
    ):
        if measure == "cosine":
            similarity = cosine_distance
        else:
            similarity = euclidean_distance
        collect_label = []
        for i in range(collect_data.shape[0]):
            sim = [
                similarity(collect_data[i], known_data[j])
                for j in range(known_data.shape[0])
            ]
            collect_label.append(known_label[np.argmin(sim)])
        return np.array(collect_label)

    return distance_attack


def kmeans_based(collect_data: np.array, known_data: np.array, known_labels: dict):

    unique_labels = np.unique(known_labels)
    n_clusters = len(unique_labels)

    known_dict = {}
    for label in unique_labels:
        known_dict[label] = known_data[known_labels == label]

    init_centroids = []
    for label in unique_labels:
        centroid = known_dict[label][np.random.choice(len(known_dict[label]))]
        init_centroids.append(centroid)

    kmeans = KMeans(n_clusters=n_clusters, init=np.array(init_centroids), n_init=1)
    kmeans.fit(collect_data)

    cluster_labels = kmeans.labels_
    known_cluster_labels = kmeans.predict(known_data)
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(known_labels)):
        true_label_idx = np.where(unique_labels == known_labels[i])[0][0]
        pred_label_idx = known_cluster_labels[i]
        confusion_matrix[true_label_idx, pred_label_idx] += 1
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    cluster_to_label_map = {
        col_ind[i]: unique_labels[row_ind[i]] for i in range(len(row_ind))
    }
    pred_labels = np.array(
        [cluster_to_label_map.get(label, -1) for label in cluster_labels]
    )
    return pred_labels


def random_known_data(known_data, known_label, num=2):
    known_data_list = []
    known_label_list = []
    for i in np.unique(known_label):
        idx = np.where(known_label == i)[0]
        idx = np.random.choice(idx, num, replace=False)
        known_data_list.append(known_data[idx])
        known_label_list.append(known_label[idx])
    known_data = np.concatenate(known_data_list, axis=0)
    known_label = np.concatenate(known_label_list, axis=0)
    return known_data, known_label


def get_attack_method(attack_method, measure="cosine"):
    if attack_method == "distance":
        return distance_based(measure=measure)
    elif attack_method == "k-means":
        return kmeans_based
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")


class SimilarityLabelInferenceAttack(AttackCallback):
    """
    Implementation of sim_lia aglorithm in paper: Similarity-based Label Inference Attack against Training and Inference of Split Learning
    Attributes:
        attack_party: The party that performs the attack.
        label_party: The party that has the labels.
        data_type: The type of data to be used for the attack. Can be "feature" or "grad".
        attack_method: The method used for the attack. Can be "distance" or "k-means".
        known_num: The number of known data points to be used for the attack.
        distance_metric: The distance metric to be used for the attack. Can be "cosine" or "euclidean".
        exec_device: The device to be used for execution. Can be "cpu" or "gpu".
        params: Additional parameters for the attack.
    """

    def __init__(
        self,
        attack_party: PYU,
        label_party: PYU,
        data_type: str,
        attack_method: str,
        known_num: int,
        distance_metric="cosine",
        exec_device="cpu",
        **params,
    ):
        super().__init__(
            **params,
        )

        self.attack_party = attack_party
        self.label_party = label_party

        self.data_type = data_type
        self.attack_method = attack_method
        self.distance_metric = distance_metric
        self.known_num = known_num
        self.exec_device = exec_device

        if self.attack_method not in ["distance", "k-means"]:
            raise ValueError(
                f"attack_method should be distance or k-means, but got {self.attack_method}"
            )

        if self.attack_method == "distance":
            self.known_num = 1

        self.attack_func = get_attack_method(self.attack_method, self.distance_metric)

        self.last_epoch = False

        self.res = None
        self.metrics = None

    def on_epoch_begin(self, epoch=None, logs=None):
        # The attacker uses the method during the final epoch.
        if epoch == self.params["epochs"] - 1:
            self.last_epoch = True

        def prepare_store(worker):
            if self.last_epoch:
                worker._callback_store["data"] = []

        self._workers[self.attack_party].apply(prepare_store)

    def on_base_forward_end(self):
        # record hidden state value
        if self.data_type == "feature":

            def record_data(worker):
                if (
                    worker.model_base.training
                    and worker._callback_store.get("data", None) is not None
                ):
                    if isinstance(worker._h, list):
                        data = worker._h[0].cpu().detach().numpy()
                    else:
                        data = worker._h.cpu().detach().numpy()
                    worker._callback_store["data"].append(data)

            self._workers[self.attack_party].apply(record_data)

    def on_base_backward_end(self):
        # record grad value
        if self.data_type == "grad":

            def record_data(worker):
                if (
                    worker.model_base.training
                    and worker._callback_store.get("data", None) is not None
                ):
                    grad = worker._gradient.cpu().numpy()
                    worker._callback_store["data"].append(grad)

            self._workers[self.attack_party].apply(record_data)

    def on_fuse_forward_begin(self):
        if self.last_epoch:

            def get_label(worker):
                if isinstance(worker.train_y, list):
                    label = worker.train_y[0].cpu().detach().numpy()
                else:
                    label = worker.train_y.cpu().detach().numpy()
                return label

            label = self._workers[self.label_party].apply(get_label)

            def store_label(worker, label):
                if worker._callback_store.get("labels", None) is None:
                    worker._callback_store["labels"] = []
                worker._callback_store["labels"].append(label)

            # reveal the data due to the evaluate
            label = reveal(label)
            self._workers[self.attack_party].apply(store_label, label)

    def on_train_end(self, logs=None):

        def label_inference_attack(worker):
            data = np.concatenate(worker._callback_store["data"], axis=0)
            labels = np.concatenate(worker._callback_store["labels"], axis=0)
            data = preprocessing.normalize(data)
            data = data.reshape((data.shape[0], -1))

            known_data, known_label = random_known_data(data, labels, self.known_num)
            result_label = self.attack_func(data, known_data, known_label)
            res = {"attack_acc": sum(result_label == labels) / len(labels)}
            return res

        res = self._workers[self.attack_party].apply(label_inference_attack)
        wait(res)
        self.metrics = reveal(res)

    def get_attack_metrics(self):
        return self.metrics
