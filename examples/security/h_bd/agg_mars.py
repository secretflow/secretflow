# Copyright 2025 Ant Group Co., Ltd.
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


from copy import deepcopy
from typing import List
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance

from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.device import PYU, DeviceObject, PYUObject


def CLP(params_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute Lipschitz Constant Vector for model parameters (CLP)

    Processing logic:
    1. Handle convolutional, BN, and fully-connected layers separately
    2. Calculate layer-wise Lipschitz constants using SVD
    3. Maintain compatibility with parameter list input format

    Args:
        params_list: List of model parameters (numpy arrays) organized layer-wise

    Returns:
        Array of selected Lipschitz constants (top 5% values)
    """
    all_lips = []
    u = 0.05  # Top percentage threshold
    current_idx = 0

    # Create parameter type markers (conv/bn/fc)
    param_types = []
    while current_idx < len(params_list):
        param = params_list[current_idx]

        # Detect convolutional layer (4D weights)
        if param.ndim == 4:
            param_types.append(('conv', current_idx))
            current_idx += 1

        # Detect BN layer group (4 consecutive 1D params)
        elif param.ndim == 1 and current_idx + 3 < len(params_list):
            next_params = params_list[current_idx : current_idx + 4]
            if all(p.ndim == 1 for p in next_params):
                param_types.append(('bn', current_idx))
                current_idx += 4
            else:
                current_idx += 1

        # Detect fully-connected layer (2D weights)
        elif param.ndim == 2:
            param_types.append(('fc', current_idx))
            current_idx += 1

        else:
            current_idx += 1

    # Process each layer based on detected types
    for i, (ptype, idx) in enumerate(param_types):
        if ptype == 'conv':
            # ------ Convolutional Layer Processing ------
            conv_weight = params_list[idx]

            # Find subsequent BN parameters
            bn_weight = bn_bias = bn_var = None
            if i + 1 < len(param_types) and param_types[i + 1][0] == 'bn':
                bn_idx = param_types[i + 1][1]
                bn_weight = params_list[bn_idx]
                bn_bias = params_list[bn_idx + 1]
                bn_var = params_list[bn_idx + 3]  # running_var at 4th position

            # Calculate per-channel Lipschitz constants
            channel_lips = []
            for ch in range(conv_weight.shape[0]):
                w = conv_weight[ch].reshape(conv_weight.shape[1], -1)

                # Apply BN scaling if available
                if bn_weight is not None and ch < bn_weight.shape[0]:
                    var = np.maximum(bn_var[ch], 0) + 1e-5
                    scale = np.abs(bn_weight[ch] / np.sqrt(var))
                    w_scaled = w * scale
                else:
                    w_scaled = w

                # Compute maximum singular value
                try:
                    s = np.linalg.svd(w_scaled, compute_uv=False)
                    lips = s[0]
                except np.linalg.LinAlgError:
                    lips = 0.0
                channel_lips.append(lips)

            # Select top u% values
            if channel_lips:
                threshold = np.quantile(channel_lips, 1 - u)
                all_lips.extend([lip for lip in channel_lips if lip >= threshold])

        elif ptype == 'bn':
            # ------ BatchNorm Layer Processing ------
            weight = params_list[idx]
            running_var = params_list[idx + 3]

            # Compute Lipschitz constants: |weight / sqrt(var + eps)|
            var = np.maximum(running_var, 0) + 1e-5
            std = np.sqrt(var)
            lips = np.abs(weight / std)

            # Select top u% values
            threshold = np.quantile(lips, 1 - u)
            all_lips.extend(lips[lips >= threshold].tolist())

        elif ptype == 'fc':
            # ------ Fully-Connected Layer Processing ------
            fc_weight = params_list[idx]

            # Compute maximum singular value
            try:
                w = fc_weight.reshape(fc_weight.shape[0], -1)
                s = np.linalg.svd(w, compute_uv=False)
                lips = s[0]
            except np.linalg.LinAlgError:
                lips = 0.0

            # Single Lipschitz value per FC layer
            threshold = np.quantile([lips], 1 - u)
            if lips >= threshold:
                all_lips.append(lips)

    # Fallback: Return parameter norms if no Lipschitz values computed
    if not all_lips:
        return np.array(
            [
                np.linalg.norm(p.ravel())
                for p in params_list
                if isinstance(p, np.ndarray)
            ]
        )
    return np.array(all_lips)


# -------------------------------
# Helper Functions
# -------------------------------


def compute_wasserstein_distance_matrix(l):
    """Compute pairwise Wasserstein distance matrix between feature vectors

    Args:
        l: List of feature vectors (Lipschitz constant arrays)

    Returns:
        NxN distance matrix
    """
    n = len(l)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Handle empty features
            if len(l[i]) == 0 or len(l[j]) == 0:
                distance = 0.0
            else:
                # Create normalized histograms
                hist1, _ = np.histogram(l[i], bins=50, density=True)
                hist2, _ = np.histogram(l[j], bins=50, density=True)
                distance = wasserstein_distance(hist1, hist2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


def kmeans_wasserstein_clustering(l, k=2):
    """Perform KMeans clustering using Wasserstein distance matrix"""
    distance_matrix = compute_wasserstein_distance_matrix(l)
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(distance_matrix)


def compute_cluster_centers(l, labels, k):
    """Compute cluster centers using histogram aggregation"""
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].append(l[i])

    centers = []
    for cluster in clusters:
        if not cluster:
            centers.append(np.array([]))
            continue

        # Combine samples and create representative histogram
        combined = np.concatenate(cluster)
        hist, _ = np.histogram(combined, bins=50, density=True)
        centers.append(hist)

    return centers


def detect_anomaly(l, kmeans, threshold):
    """Identify anomalous clusters based on center distances"""
    labels = kmeans.labels_
    centers = compute_cluster_centers(l, labels, k=2)

    # Filter empty clusters
    valid_centers = [c for c in centers if len(c) > 0]
    if len(valid_centers) < 2:
        return np.zeros(len(labels), dtype=int)

    # Calculate inter-cluster distance
    center_distance = wasserstein_distance(valid_centers[0], valid_centers[1])

    if center_distance <= threshold:
        return np.zeros(len(labels), dtype=int)  # No anomalies
    else:
        # Identify anomalous cluster (higher norm)
        center_norms = [np.linalg.norm(center, ord=1) for center in valid_centers]
        anomaly_cluster = np.argmax(center_norms)
        return np.where(labels == anomaly_cluster, -1, 0)


def cluster_and_detect_anomalies(l, threshold):
    """Full anomaly detection pipeline"""
    if len(l) < 2:
        return np.zeros(len(l), dtype=int)

    try:
        kmeans = kmeans_wasserstein_clustering(l)
        return detect_anomaly(l, kmeans, threshold)
    except Exception as e:
        print(f"Clustering failed: {e}")
        return np.zeros(len(l), dtype=int)


# -------------------------------
# MARS Aggregator Implementation
# -------------------------------


class MarsAggregator(PlainAggregator):
    """
    Byzantine-Robust Aggregator using Lipschitz-based Anomaly Detection

    Features:
    - Processes parameter lists organized by layer (numpy arrays)
    - Automatically detects Conv-BN parameter groups
    - Performs Wasserstein-based anomaly detection
    - Implements secure aggregation
    """

    def __init__(self, device: PYU, anomaly_threshold=1.0):
        assert isinstance(device, PYU), f"Requires PYU device but got {type(device)}"
        self.device = device
        self.anomaly_threshold = anomaly_threshold

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
        # Data preparation
        data = [d.to(self.device) for d in data]
        weights = (
            [
                w.to(self.device) if isinstance(w, DeviceObject) else (w or 1.0)
                for w in weights
            ]
            if weights
            else None
        )

        def _average(*data, axis, weights):
            num_clients = len(data)

            # 1. Extract Lipschitz features
            lips_vectors = []
            for client_params in data:
                try:
                    lips = CLP(list(client_params))
                    lips_vectors.append(lips)
                except Exception as e:
                    print(f"CLP computation failed: {e}")
                    lips_vectors.append(np.array([]))

            # 2. Anomaly detection
            anomaly_labels = cluster_and_detect_anomalies(
                lips_vectors, self.anomaly_threshold
            )
            print(f"Anomaly detection results: {anomaly_labels}")

            # 3. Filter anomalous clients
            benign_clients = []
            benign_weights = []
            for i in range(num_clients):
                if anomaly_labels[i] != -1:
                    benign_clients.append(data[i])
                    benign_weights.append(weights[i] if weights else 1.0)

            # 4. Secure aggregation
            if not benign_clients:
                print(
                    "Warning: All clients marked as anomalous, returning first client's parameters"
                )
                return deepcopy(data[0])

            # Layer-wise aggregation
            aggregated_params = []
            num_layers = len(benign_clients[0])
            for layer_idx in range(num_layers):
                layer_values = [client[layer_idx] for client in benign_clients]

                # Weighted average with fallback
                try:
                    avg = np.average(layer_values, axis=0, weights=benign_weights)
                except:
                    print(f"Layer {layer_idx} aggregation failed, using median")
                    avg = np.median(layer_values, axis=0)

                aggregated_params.append(avg)

            return aggregated_params

        return self.device(_average)(*data, axis=axis, weights=weights)
