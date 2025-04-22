# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from gensim.models.word2vec import Word2Vec
from DataSet import DataSet
import pickle


def k_means(num_clusters, embeddings):
    """
    Perform KMeans clustering and return cluster centers for each sample and their labels.
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Assign each sample to its corresponding cluster centroid
    centroids_matrix = np.zeros_like(embeddings)
    for i in range(embeddings.shape[0]):
        cluster_label = cluster_labels[i]
        centroids_matrix[i] = centroids[cluster_label]

    return centroids_matrix, cluster_labels


if __name__ == '__main__':
    dataset_names = ["movie", "music", "book"]
    num_clusters = 64
    k_size = 8

    for dataset_name in dataset_names:
        dataset = DataSet(dataset_name, None)
        num_users = dataset.shape[0]
        num_items = dataset.shape[1]

        model = Word2Vec.load(f"Node2vec_{dataset_name}_KSize_{k_size}.model")
        node_vectors = torch.tensor(model.wv.vectors, dtype=torch.float32)

        all_node_keys = [str(i) for i in range(num_users + num_items)]
        valid_indices = [model.wv.key_to_index[key]
                         for key in all_node_keys if key in model.wv.key_to_index]
        node_vectors = node_vectors[valid_indices]

        user_indices = np.arange(num_users)
        item_indices = np.arange(num_users, num_users + num_items)
        user_vectors = node_vectors[user_indices]
        item_vectors = node_vectors[item_indices]

        # Standardize features
        scaler = StandardScaler()
        user_vectors_scaled = scaler.fit_transform(user_vectors)
        item_vectors_scaled = scaler.transform(item_vectors)

        # Clustering
        user_centroids, user_labels = k_means(num_clusters, user_vectors)
        item_centroids, item_labels = k_means(num_clusters, item_vectors)

        # Save clustering results
        results = {
            'user_centroids': user_centroids,
            'user_labels': user_labels,
            'item_centroids': item_centroids,
            'item_labels': item_labels
        }
        with open(f'clustering_results_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(
            f"Clustering results for {dataset_name} saved as clustering_results_{dataset_name}.pkl")
