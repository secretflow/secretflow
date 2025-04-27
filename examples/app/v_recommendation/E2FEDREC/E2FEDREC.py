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
import torch.nn as nn
import torch.nn.functional as F
import math
import heapq
import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
import pandas as pd
import logging

logging.getLogger("gensim").setLevel(logging.WARNING)


class Model(nn.Module):
    """
    Single-domain (Domain A) recommendation model enhanced with
    cross-domain contrastive learning based on similarity to Domain B.
    """

    def __init__(self, args, dataName, dataName_B, shape, maxRate, device_id=0):
        super(Model, self).__init__()
        # Select computation device (GPU preferred if available)
        if torch.cuda.is_available():
            self.device_id = device_id
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f"cuda:{self.device_id}")
        else:
            self.device_id = -1
            self.device = torch.device("cpu")

        # Dataset and training configurations
        self.dataName = dataName  # Dataset name for Domain A
        self.dataName_B = (
            dataName_B  # Dataset name for Domain B (for cross-domain tasks)
        )
        self.KSize = args.KSize  # Embedding dimension size (output of Node2Vec)

        # Load pre-trained Node2Vec embeddings for Domain A
        self.model_N2V = Word2Vec.load(
            f"Node2vec_{self.dataName}_KSize_{self.KSize}.model"
        )

        self.shape = shape  # (num_users, num_items) in Domain A
        self.maxRate = maxRate  # Maximum rating value (for normalization)
        self.lr = args.lr  # Learning rate

        # Model architecture hyperparameters
        self.userLayer = args.userLayer  # List of hidden sizes for user MLP
        self.itemLayer = args.itemLayer  # List of hidden sizes for item MLP
        self.lambdad = args.lambdad  # L2 regularization weight
        self.ssl_temp = args.ssl_temp  # Temperature parameter for contrastive loss
        self.ssl_reg_intra = args.ssl_reg_intra  # Intra-domain SSL loss weight
        self.ssl_reg_inter = args.ssl_reg_inter  # Inter-domain SSL loss weight

        # Load model inputs and auxiliary structures
        self._load_node_features()
        self._load_auxiliary_features()
        self._load_cross_domain_similarity()

        # Build neural network layers and optimizer
        self.build_model()

    def _load_node_features(self):
        """Load Node2Vec pre-trained node embeddings for Domain A."""
        self.node_features = torch.tensor(
            self.model_N2V.wv.vectors, dtype=torch.float32
        ).to(self.device)

        # Filter node features to keep only valid users and items
        user_item_keys = [str(i) for i in range(self.shape[0] + self.shape[1])]
        valid_indices = [
            self.model_N2V.wv.vocab[key].index
            for key in user_item_keys
            if key in self.model_N2V.wv.vocab
        ]
        self.node_features = self.node_features[valid_indices]

    def _load_auxiliary_features(self):
        """Load clustering centroids (auxiliary features) for Domain A."""
        with open(f"./pkl/clustering_results_{self.dataName}.pkl", "rb") as f:
            centroids = pickle.load(f).get("centroids_matrix", None)
            self.centroids_matrix = torch.tensor(centroids, dtype=torch.float32).to(
                self.device
            )

    def _load_cross_domain_similarity(self):
        """
        Load cross-domain similarity embeddings from Domain A to Domain B.

        Only A-to-B mappings are used; B-to-A mappings are ignored in this model.
        """
        with open(
            f"./pkl/{self.dataName}_to_{self.dataName_B}_similar_vectors.pkl", "rb"
        ) as f:
            similarity_vectors = pickle.load(f).get("vectors_B", None)
            self.to_B_similarity = torch.tensor(
                similarity_vectors, dtype=torch.float32
            ).to(self.device)

        self.from_B_similarity = (
            None  # Reserved if B-to-A similarities are needed later
        )

    def build_model(self):
        """
        Build the encoder architecture for users and items in Domain A.

        - Initialize first projection layers (linear transforms from Node2Vec to first hidden layer)
        - Build stacked MLPs for users and items
        - Setup optimizer
        """
        import torch.nn.init as init

        # First layer projections
        self.user_W1 = nn.Parameter(torch.empty(self.KSize, self.userLayer[0]))
        self.item_W1 = nn.Parameter(torch.empty(self.KSize, self.itemLayer[0]))
        init.kaiming_uniform_(self.user_W1, a=math.sqrt(5))
        init.kaiming_uniform_(self.item_W1, a=math.sqrt(5))

        # Multi-layer perceptrons for user and item embeddings
        self.user_layers = nn.ModuleList(
            [
                nn.Linear(self.userLayer[i], self.userLayer[i + 1])
                for i in range(len(self.userLayer) - 1)
            ]
        )
        self.item_layers = nn.ModuleList(
            [
                nn.Linear(self.itemLayer[i], self.itemLayer[i + 1])
                for i in range(len(self.itemLayer) - 1)
            ]
        )

        # Define optimizer (Adam)
        params = (
            [self.user_W1, self.item_W1]
            + list(self.user_layers.parameters())
            + list(self.item_layers.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        # Move model to target device
        self.to(self.device)

    def forward(self, user, item):
        """
        Forward pass to compute user and item embeddings.

        Args:
            user: Tensor of user indices (batch)
            item: Tensor of item indices (batch)

        Returns:
            user_out: Final user embeddings
            item_out: Final item embeddings
        """
        # Node features are shared for both users and items
        node_features = self.node_features
        shape = self.shape

        # Encode users
        user_input = node_features[user]
        user_out = torch.matmul(user_input, self.user_W1)
        for layer in self.user_layers:
            user_out = torch.relu(layer(user_out))

        # Encode items
        item_input = node_features[shape[0] + item]
        item_out = torch.matmul(item_input, self.item_W1)
        for layer in self.item_layers:
            item_out = torch.relu(layer(item_out))

        return user_out, item_out

    def add_loss_main(self, u_embeddings, i_embeddings, rate, maxRate):
        """
        Compute the primary recommendation loss (binary cross-entropy) or prediction scores.

        Args:
            u_embeddings: User embedding tensor
            i_embeddings: Item embedding tensor
            rate: Rating tensor (if None, only compute scores)
            maxRate: Maximum rating value for normalization

        Returns:
            If rate is None: Tensor of predicted scores
            Else: (loss value, predicted scores)
        """
        # Cosine similarity as predicted relevance
        norm_user_output = torch.norm(u_embeddings, dim=1)
        norm_item_output = torch.norm(i_embeddings, dim=1)
        dot_product = torch.sum(u_embeddings * i_embeddings, dim=1)
        predict = dot_product / (norm_user_output * norm_item_output + 1e-8)
        predict = torch.clamp(predict, min=1e-6)  # Prevent numerical instability

        if rate is None:
            return predict

        # Binary cross-entropy loss for recommendation
        regRate = rate / maxRate
        loss = -torch.sum(
            regRate * torch.log(predict) + (1 - regRate) * torch.log(1 - predict)
        )

        # L2 regularization on embeddings
        loss += self.lambdad * (u_embeddings.norm(p=2) + i_embeddings.norm(p=2))

        return loss, predict

    def calc_ssl_loss_inter(
        self, u_embeddings, node_features, u_side_embeddings, user_idx, shape
    ):
        """
        Compute inter-domain contrastive loss between user embeddings and cross-domain similar users.

        Args:
            u_embeddings: Current domain's user embeddings
            node_features: Node features of the current domain
            u_side_embeddings: Cross-domain user embeddings from domain B
            user_idx: Indices of current users
            shape: (num_users, num_items)

        Returns:
            Inter-domain contrastive loss value
        """
        # Fetch corresponding similar users from domain B
        emb2 = u_side_embeddings[user_idx]

        # Positive pair: same user across domains
        pos_score = torch.sum(
            F.normalize(u_embeddings, dim=1) * F.normalize(emb2, dim=1), dim=1
        )

        # Negative pairs: similarity to all users
        ttl_score = F.normalize(u_embeddings, dim=1) @ F.normalize(
            node_features[0 : shape[0]].T, dim=0
        )

        # InfoNCE loss for cross-domain contrastive learning
        ssl_loss = -torch.sum(
            torch.log(torch.exp(pos_score / self.ssl_temp))
            / torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
        )

        return self.ssl_reg_inter * ssl_loss

    def calc_ssl_loss_intra(self, u_embeddings, cluster_feature, shape, user):
        """
        Compute intra-domain contrastive loss between users and their cluster centroids.

        Args:
            u_embeddings: User embedding tensor
            cluster_feature: Cluster centroid tensor
            shape: (num_users, num_items)
            user: Tensor of user indices

        Returns:
            Intra-domain contrastive loss value
        """
        # Normalize user embeddings and cluster centroids
        anchor = F.normalize(u_embeddings, p=2, dim=1)
        positive = F.normalize(cluster_feature[user], p=2, dim=1)
        all_pos = F.normalize(cluster_feature[0 : shape[0]], p=2, dim=1)

        # Calculate logits for all possible positives
        logits = torch.matmul(anchor, all_pos.T) / self.ssl_temp

        # Construct ground-truth mask for positive samples
        pos_mask = torch.zeros_like(logits).scatter_(1, user.unsqueeze(1), 1.0)

        # Cross-entropy loss (InfoNCE style)
        log_probs = F.log_softmax(logits, dim=1)
        ssl_loss_user = -torch.sum(pos_mask * log_probs)

        return self.ssl_reg_intra * ssl_loss_user

    def train_model(self, train_u, train_i, train_r, batch_size):
        """
        Perform one epoch of training.

        Args:
            train_u: Tensor of user indices for training
            train_i: Tensor of item indices for training
            train_r: Tensor of ratings for training
            batch_size: Mini-batch size

        Returns:
            Average training loss over all batches
        """
        self.train()
        num_batches = len(train_u) // batch_size + 1
        losses = []

        for i in range(num_batches):
            min_idx = i * batch_size
            max_idx = min(len(train_u), (i + 1) * batch_size)

            # Mini-batch slicing
            train_u_batch = train_u[min_idx:max_idx]
            train_i_batch = train_i[min_idx:max_idx]
            train_r_batch = train_r[min_idx:max_idx]

            # Forward pass and loss calculation
            self.optimizer.zero_grad()
            user_out, item_out = self.forward(train_u_batch, train_i_batch)

            loss_main, _ = self.add_loss_main(
                user_out, item_out, train_r_batch, self.maxRate
            )
            loss_inter = self.calc_ssl_loss_inter(
                user_out,
                self.node_features,
                self.from_B_similarity,
                train_u_batch,
                self.shape,
            )
            loss_intra = self.calc_ssl_loss_intra(
                user_out, self.centroids_matrix, self.shape, train_u_batch
            )

            # Total loss and backpropagation
            loss = loss_main + loss_inter + loss_intra
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def evaluate(self, topK, testUser, testItem):
        """
        Evaluate the model on test users and candidate items using HR and NDCG.

        Args:
            topK: Number of top items considered for hit and NDCG calculation
            testUser: List/Tensor of test user indices
            testItem: List of candidate item lists (each contains one positive and several negatives)

        Returns:
            (HR@K, NDCG@K) scores
        """
        self.eval()

        def getHitRatio(ranklist, targetItem):
            """Compute Hit Ratio: 1 if target item is ranked in topK, else 0."""
            return 1 if targetItem in ranklist else 0

        def getNDCG(ranklist, targetItem):
            """Compute normalized discounted cumulative gain."""
            for i, item in enumerate(ranklist):
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        hr = []
        ndcg = []

        with torch.no_grad():
            for i in range(len(testUser)):
                target = testItem[i][0]
                user_out, item_out = self.forward(testUser[i], testItem[i])
                y = self.add_loss_main(user_out, item_out, None, self.maxRate)

                # Ranking candidate items based on predicted scores
                item_score_dict = {
                    item: y[j].item() for j, item in enumerate(testItem[i])
                }
                ranklist = heapq.nlargest(
                    topK, item_score_dict, key=item_score_dict.get
                )

                # Calculate metrics
                hr.append(getHitRatio(ranklist, target))
                ndcg.append(getNDCG(ranklist, target))

        return np.mean(hr), np.mean(ndcg)
