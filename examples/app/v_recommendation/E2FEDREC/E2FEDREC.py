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
from DataSet import DataSet
import matplotlib.pyplot as plt
import pandas as pd
import sys
from secretflow import PYUObject, proxy
import logging
logging.getLogger('gensim').setLevel(logging.WARNING)


class Model(nn.Module):
    """
    A dual-domain recommendation model with cross-domain and intra-domain contrastive learning.
    The model learns user and item representations from two domains (A and B) simultaneously,
    leveraging node2vec embeddings and clustering information for enhanced performance.
    """

    def __init__(self, args, shape_A, shape_B, maxRate_A, maxRate_B, device_id=0):
        super(Model, self).__init__()
    # 自动判断是否有可用 GPU
        if torch.cuda.is_available():
            self.device_id = device_id
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f'cuda:{self.device_id}')
        else:
            self.device_id = -1
            self.device = torch.device('cpu')

        # Domain configurations
        self.dataName_A = args.dataName_A
        self.dataName_B = args.dataName_B
        self.KSize = args.KSize

        # Load pre-trained node2vec models for both domains
        self.model_N2V_A = Word2Vec.load("Node2vec_" + self.dataName_A +
                                         "_KSize_" + str(self.KSize) + ".model")
        self.model_N2V_B = Word2Vec.load("Node2vec_" + self.dataName_B +
                                         "_KSize_" + str(self.KSize) + ".model")

        # Dataset parameters
        self.shape_A = shape_A
        self.shape_B = shape_B
        self.maxRate_A = maxRate_A
        self.maxRate_B = maxRate_B
        self.lr = args.lr

        # Model configurations
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.lambdad = args.lambdad
        self.ssl_temp = args.ssl_temp
        self.ssl_reg_intra = args.ssl_reg_intra
        self.ssl_reg_inter = args.ssl_reg_inter

        # Load node features and auxiliary data
        self._load_node_features()
        self._load_auxiliary_features()

        # Build model architecture
        self.build_model()

    def _load_node_features(self):
        """Load and process node features from node2vec models for both domains."""
        # Domain A node features
        self.node_features_A = torch.tensor(
            self.model_N2V_A.wv.vectors, dtype=torch.float32).to(self.device)
        user_keys_A = [str(i)
                       for i in range(self.shape_A[0] + self.shape_A[1])]
        valid_indices_A = [self.model_N2V_A.wv.vocab[key].index for key in user_keys_A
                           if key in self.model_N2V_A.wv.vocab]
        self.node_features_A = self.node_features_A[valid_indices_A]

        # Domain B node features
        self.node_features_B = torch.tensor(
            self.model_N2V_B.wv.vectors, dtype=torch.float32).to(self.device)
        user_keys_B = [str(i)
                       for i in range(self.shape_B[0] + self.shape_B[1])]
        valid_indices_B = [self.model_N2V_B.wv.vocab[key].index for key in user_keys_B
                           if key in self.model_N2V_B.wv.vocab]
        self.node_features_B = self.node_features_B[valid_indices_B]

    def _load_auxiliary_features(self):
        """Load clustering centroids and cross-domain similarity vectors."""
        # Domain A clustering centroids
        with open(f'./pkl/clustering_results_{self.dataName_A}.pkl', 'rb') as f:
            self.centroids_matrix_A = pickle.load(f)
            self.centroids_matrix_A = self.centroids_matrix_A.get(
                'centroids_matrix', None)
            self.centroids_matrix_A = torch.tensor(
                self.centroids_matrix_A, dtype=torch.float32).to(self.device)

        # Domain B clustering centroids
        with open(f'./pkl/clustering_results_{self.dataName_B}.pkl', 'rb') as f:
            self.centroids_matrix_B = pickle.load(f)
            self.centroids_matrix_B = self.centroids_matrix_B.get(
                'centroids_matrix', None)
            self.centroids_matrix_B = torch.tensor(
                self.centroids_matrix_B, dtype=torch.float32).to(self.device)

        # Cross-domain similarity vectors (A to B)
        with open(f'./pkl/{self.dataName_A}_to_{self.dataName_B}_similar_vectors.pkl', 'rb') as f:
            self.A_B_similarity = pickle.load(f)
            self.A_B_similarity = self.A_B_similarity.get('vectors_B', None)
            self.A_B_similarity = torch.tensor(
                self.A_B_similarity, dtype=torch.float32).to(self.device)

        # Cross-domain similarity vectors (B to A)
        with open(f'./pkl/{self.dataName_B}_to_{self.dataName_A}_similar_vectors.pkl', 'rb') as f:
            self.B_A_similarity = pickle.load(f)
            self.B_A_similarity = self.B_A_similarity.get('vectors_B', None)
            self.B_A_similarity = torch.tensor(
                self.B_A_similarity, dtype=torch.float32).to(self.device)

    def build_model(self):
        """Build the model architecture including encoders and optimizers."""
        import torch.nn.init as init

        # Initialize first layer weights with Kaiming initialization (good for ReLU)
        self.user_W1_A = nn.Parameter(
            torch.empty(self.KSize, self.userLayer[0]))
        init.kaiming_uniform_(self.user_W1_A, a=math.sqrt(5))

        self.user_W1_B = nn.Parameter(
            torch.empty(self.KSize, self.userLayer[0]))
        init.kaiming_uniform_(self.user_W1_B, a=math.sqrt(5))

        self.item_W1_A = nn.Parameter(
            torch.empty(self.KSize, self.itemLayer[0]))
        init.kaiming_uniform_(self.item_W1_A, a=math.sqrt(5))

        self.item_W1_B = nn.Parameter(
            torch.empty(self.KSize, self.itemLayer[0]))
        init.kaiming_uniform_(self.item_W1_B, a=math.sqrt(5))

        # User encoders for both domains
        self.user_layers_A = nn.ModuleList([
            nn.Linear(self.userLayer[i], self.userLayer[i + 1])
            for i in range(len(self.userLayer) - 1)
        ])
        self.user_layers_B = nn.ModuleList([
            nn.Linear(self.userLayer[i], self.userLayer[i + 1])
            for i in range(len(self.userLayer) - 1)
        ])

        # Item encoders for both domains
        self.item_layers_A = nn.ModuleList([
            nn.Linear(self.itemLayer[i], self.itemLayer[i + 1])
            for i in range(len(self.itemLayer) - 1)
        ])
        self.item_layers_B = nn.ModuleList([
            nn.Linear(self.itemLayer[i], self.itemLayer[i + 1])
            for i in range(len(self.itemLayer) - 1)
        ])

        # Separate optimizers for each domain
        params_A = [self.user_W1_A, self.item_W1_A] + \
            list(self.user_layers_A.parameters()) + \
            list(self.item_layers_A.parameters())
        self.optimizer_A = torch.optim.Adam(params_A, lr=self.lr)

        params_B = [self.user_W1_B, self.item_W1_B] + \
            list(self.user_layers_B.parameters()) + \
            list(self.item_layers_B.parameters())
        self.optimizer_B = torch.optim.Adam(params_B, lr=self.lr)

        # Move entire model to specified device
        self.to(self.device)

    def forward(self, user, item, domain='A'):
        """
        Forward pass for the model.

        Args:
            user: Tensor of user indices
            item: Tensor of item indices
            domain: Which domain to process ('A' or 'B')

        Returns:
            user_out: User embeddings
            item_out: Item embeddings
        """
        if domain == 'A':
            node_features = self.node_features_A
            shape = self.shape_A
            user_W1 = self.user_W1_A
            item_W1 = self.item_W1_A
            user_layers = self.user_layers_A
            item_layers = self.item_layers_A
        elif domain == 'B':
            node_features = self.node_features_B
            shape = self.shape_B
            user_W1 = self.user_W1_B
            item_W1 = self.item_W1_B
            user_layers = self.user_layers_B
            item_layers = self.item_layers_B
        else:
            raise ValueError("Invalid domain. Please use 'A' or 'B'.")

        # User embedding forward pass
        user_input = node_features[user]
        user_out = torch.matmul(user_input, user_W1)
        for layer in user_layers:
            user_out = torch.relu(layer(user_out))

        # Item embedding forward pass (note: items are stored after users in node features)
        item_input = node_features[shape[0] + item]
        item_out = torch.matmul(item_input, item_W1)
        for layer in item_layers:
            item_out = torch.relu(layer(item_out))

        return user_out, item_out

    def add_loss_main(self, u_embeddings, i_embeddings, rate, maxRate):
        """
        Calculate the main recommendation loss (binary cross-entropy).

        Args:
            u_embeddings: User embeddings
            i_embeddings: Item embeddings
            rate: Ground truth ratings (None for evaluation)
            maxRate: Maximum rating value for normalization

        Returns:
            If rate is None: predicted scores
            Else: (loss, predicted scores)
        """
        # Calculate cosine similarity between users and items
        norm_user_output = torch.norm(u_embeddings, dim=1)
        norm_item_output = torch.norm(i_embeddings, dim=1)
        dot_product = torch.sum(u_embeddings * i_embeddings, dim=1)
        predict = dot_product / (norm_user_output * norm_item_output + 1e-8)
        predict = torch.clamp(predict, min=1e-6)  # Avoid numerical instability

        if rate is None:
            return predict

        # Normalize ratings and calculate binary cross-entropy loss
        regRate = rate / maxRate
        loss = -torch.sum(regRate * torch.log(predict) +
                          (1 - regRate) * torch.log(1 - predict))
        loss += self.lambdad * (u_embeddings.norm(p=2) +
                                i_embeddings.norm(p=2))  # L2 regularization

        return loss, predict

    def calc_ssl_loss_inter(self, u_embeddings, node_features, u_side_embeddings, user_idx, shape):
        """
        Calculate inter-domain contrastive loss (user embeddings vs cross-domain similar users).

        Args:
            u_embeddings: User embeddings from current domain
            node_features: All node features from current domain
            u_side_embeddings: Similar user embeddings from other domain
            user_idx: Indices of current users
            shape: (num_users, num_items) of current domain

        Returns:
            Inter-domain contrastive loss
        """
        emb2 = u_side_embeddings[user_idx]  # Get corresponding similar users from other domain

        # Positive scores: similarity between user and their cross-domain counterpart
        pos_score = torch.sum(F.normalize(u_embeddings, dim=1)
                              * F.normalize(emb2, dim=1), dim=1)

        # Negative scores: similarity between user and all other users
        ttl_score = F.normalize(
            u_embeddings, dim=1) @ F.normalize(node_features[0:shape[0]].T, dim=0)

        # InfoNCE loss
        ssl_loss = -torch.sum(
            torch.log(torch.exp(pos_score / self.ssl_temp)) /
            torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1))

        return self.ssl_reg_inter * ssl_loss

    def calc_ssl_loss_intra(self, u_embeddings, cluster_feature, shape, user):
        """
        Calculate intra-domain contrastive loss (user embeddings vs cluster centroids).

        Args:
            u_embeddings: User embeddings
            cluster_feature: Cluster centroids
            shape: (num_users, num_items)
            user: Indices of current users

        Returns:
            Intra-domain contrastive loss
        """
        # Normalize embeddings and cluster features
        anchor = F.normalize(u_embeddings, p=2, dim=1)                # [B, D]
        positive = F.normalize(cluster_feature[user], p=2, dim=1)     # [B, D]
        all_pos = F.normalize(
            cluster_feature[0:shape[0]], p=2, dim=1)  # [N_users, D]

        # Calculate similarity scores
        logits = torch.matmul(anchor, all_pos.T)  # [B, N_users]
        logits /= self.ssl_temp  # Apply temperature scaling

        # Create one-hot labels for positive pairs
        pos_mask = torch.zeros_like(logits).scatter_(1, user.unsqueeze(1), 1.0)

        # Calculate InfoNCE loss
        log_probs = F.log_softmax(logits, dim=1)
        ssl_loss_user = -torch.sum(pos_mask * log_probs)

        return self.ssl_reg_intra * ssl_loss_user

    def train_model(self, train_u_A, train_i_A, train_r_A, train_u_B, train_i_B, train_r_B, batch_size):
        """
        Train the model for one epoch.

        Args:
            train_u_A: User indices for domain A
            train_i_A: Item indices for domain A
            train_r_A: Ratings for domain A
            train_u_B: User indices for domain B
            train_i_B: Item indices for domain B
            train_r_B: Ratings for domain B
            batch_size: Training batch size

        Returns:
            Average losses for both domains
        """
        self.train()
        num_batches_A = len(train_u_A) // batch_size + 1
        num_batches_B = len(train_u_B) // batch_size + 1

        losses_A = []
        losses_B = []
        max_num_batches = max(num_batches_A, num_batches_B)

        for i in range(max_num_batches):
            min_idx = i * batch_size
            max_idx_A = min(len(train_u_A), (i + 1) * batch_size)
            max_idx_B = min(len(train_u_B), (i + 1) * batch_size)

            # Process domain A batch if available
            if min_idx < len(train_u_A):
                train_u_batch_A = train_u_A[min_idx:max_idx_A]
                train_i_batch_A = train_i_A[min_idx:max_idx_A]
                train_r_batch_A = train_r_A[min_idx:max_idx_A]

                self.optimizer_A.zero_grad()
                user_out, item_out = self.forward(
                    train_u_batch_A, train_i_batch_A, domain='A')

                # Calculate main recommendation loss
                loss_main_A, _ = self.add_loss_main(
                    user_out, item_out, train_r_batch_A, self.maxRate_A)

                # Calculate inter-domain contrastive loss
                loss_inter_A = self.calc_ssl_loss_inter(
                    user_out, self.node_features_A, self.A_B_similarity,
                    train_u_batch_A, self.shape_A)

                loss_intra_A = self.calc_ssl_loss_intra(
                    user_out, self.centroids_matrix_A, self.shape_A, train_u_batch_A)

                # Total loss and backpropagation
                loss_A = loss_main_A + loss_inter_A + loss_intra_A
                loss_A.backward()
                self.optimizer_A.step()
                losses_A.append(loss_A.item())

            # Process domain B batch if available
            if min_idx < len(train_u_B):
                train_u_batch_B = train_u_B[min_idx:max_idx_B]
                train_i_batch_B = train_i_B[min_idx:max_idx_B]
                train_r_batch_B = train_r_B[min_idx:max_idx_B]

                self.optimizer_B.zero_grad()
                user_out, item_out = self.forward(
                    train_u_batch_B, train_i_batch_B, domain='B')

                # Calculate main recommendation loss
                loss_main_B, _ = self.add_loss_main(
                    user_out, item_out, train_r_batch_B, self.maxRate_B)

                # Calculate inter-domain contrastive loss
                loss_inter_B = self.calc_ssl_loss_inter(
                    user_out, self.node_features_B, self.B_A_similarity,
                    train_u_batch_B, self.shape_B)

                loss_intra_B = self.calc_ssl_loss_intra(
                    user_out, self.centroids_matrix_B, self.shape_B, train_u_batch_B)

                # Total loss and backpropagation
                loss_B = loss_main_B + loss_inter_B
                loss_B.backward()
                self.optimizer_B.step()
                losses_B.append(loss_B.item())

        return np.mean(losses_A), np.mean(losses_B)

    def evaluate(self, topK, testUser_A, testItem_A, testUser_B, testItem_B):
        """
        Evaluate the model on test data.

        Args:
            topK: K for top-K evaluation metrics
            testUser_A: Test user indices for domain A
            testItem_A: Test item indices for domain A
            testUser_B: Test user indices for domain B
            testItem_B: Test item indices for domain B

        Returns:
            HR and NDCG for both domains
        """
        self.eval()

        def getHitRatio(ranklist, targetItem):
            """Calculate Hit Ratio (1 if target in topK, else 0)."""
            return 1 if targetItem in ranklist else 0

        def getNDCG(ranklist, targetItem):
            """Calculate Normalized Discounted Cumulative Gain."""
            for i, item in enumerate(ranklist):
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        hr_A = []
        NDCG_A = []
        hr_B = []
        NDCG_B = []

        with torch.no_grad():
            # Evaluate domain A
            for i in range(len(testUser_A)):
                target = testItem_A[i][0]
                user_out, item_out = self.forward(
                    testUser_A[i], testItem_A[i], domain='A')
                y_A = self.add_loss_main(
                    user_out, item_out, None, self.maxRate_A)

                # Create score dictionary and get top-K items
                item_score_dict = {item: y_A[j].item()
                                   for j, item in enumerate(testItem_A[i])}
                ranklist = heapq.nlargest(
                    topK, item_score_dict, key=item_score_dict.get)

                hr_A.append(getHitRatio(ranklist, target))
                NDCG_A.append(getNDCG(ranklist, target))

            # Evaluate domain B
            for i in range(len(testUser_B)):
                target = testItem_B[i][0]
                user_out, item_out = self.forward(
                    testUser_B[i], testItem_B[i], domain='B')
                y_B = self.add_loss_main(
                    user_out, item_out, None, self.maxRate_B)

                # Create score dictionary and get top-K items
                item_score_dict = {item: y_B[j].item()
                                   for j, item in enumerate(testItem_B[i])}
                ranklist = heapq.nlargest(
                    topK, item_score_dict, key=item_score_dict.get)

                hr_B.append(getHitRatio(ranklist, target))
                NDCG_B.append(getNDCG(ranklist, target))

        return np.mean(hr_A), np.mean(NDCG_A), np.mean(hr_B), np.mean(NDCG_B)


@proxy(PYUObject)
class Trainer:
    """Handles data loading, training process, and evaluation"""

    def __init__(self, args, device_id):
        self.args = args
        # 自动判断是否有可用 GPU
        if torch.cuda.is_available():
            self.device_id = device_id
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f'cuda:{self.device_id}')
        else:
            self.device_id = -1
            self.device = torch.device('cpu')

        # Load datasets
        self.dataSet_A = DataSet(args.dataName_A, None)
        self.dataSet_B = DataSet(args.dataName_B, None)

        # Initialize model
        self.model = Model(
            args,
            shape_A=self.dataSet_A.shape,
            shape_B=self.dataSet_B.shape,
            maxRate_A=self.dataSet_A.maxRate,
            maxRate_B=self.dataSet_B.maxRate,
            device_id=device_id
        )

        # Prepare test data
        self.testNeg_A = self.dataSet_A.getTestNeg(self.dataSet_A.test, 99)
        self.testNeg_B = self.dataSet_B.getTestNeg(self.dataSet_B.test, 99)

        # Training parameters
        self.batch_size = args.batchSize
        self.topK = args.topK
        self.max_epochs = args.maxEpochs
        self.negNum = args.negNum
        self.dataName_A = args.dataName_A
        self.dataName_B = args.dataName_B
        self.KSize = args.KSize

    def run(self):
        """
        Main training loop for the model.
        Handles data preparation, training, evaluation, and result saving.
        """
        # Initialize best metrics
        best_hr_A = -1
        best_NDCG_A = -1
        best_hr_B = -1
        best_NDCG_B = -1

        # Lists to track metrics over epochs
        allResults_A = []
        allResults_B = []
        loss_A_list = []
        loss_B_list = []
        hr_A_list = []
        NDCG_A_list = []
        hr_B_list = []
        NDCG_B_list = []

        print("Start Training!")

        for epoch in range(self.max_epochs):
            print(" = " * 10 + f"Epoch{epoch} " + " = " * 10)

            # Prepare domain A training data
            train_u_A, train_i_A, train_r_A = self.dataSet_A.getInstances(
                self.dataSet_A.train, self.negNum)
            train_len_A = len(train_u_A)
            shuffled_idx_A = np.random.permutation(np.arange(train_len_A))
            train_u_A = train_u_A[shuffled_idx_A]
            train_i_A = train_i_A[shuffled_idx_A]
            train_r_A = train_r_A[shuffled_idx_A]

            # Prepare domain B training data
            train_u_B, train_i_B, train_r_B = self.dataSet_B.getInstances(
                self.dataSet_B.train, self.negNum)
            train_len_B = len(train_u_B)
            shuffled_idx_B = np.random.permutation(np.arange(train_len_B))
            train_u_B = train_u_B[shuffled_idx_B]
            train_i_B = train_i_B[shuffled_idx_B]
            train_r_B = train_r_B[shuffled_idx_B]

            # Convert to tensors and move to device
            train_u_A = torch.tensor(train_u_A).to(self.device)
            train_i_A = torch.tensor(train_i_A).to(self.device)
            train_r_A = torch.tensor(train_r_A).to(self.device)
            train_u_B = torch.tensor(train_u_B).to(self.device)
            train_i_B = torch.tensor(train_i_B).to(self.device)
            train_r_B = torch.tensor(train_r_B).to(self.device)

            # Train for one epoch
            loss_A, loss_B = self.model.train_model(
                train_u_A, train_i_A, train_r_A,
                train_u_B, train_i_B, train_r_B,
                self.batch_size
            )

            print(
                f"Mean loss in this epoch is: Domain A = {loss_A}; Domain B = {loss_B}")
            print('-' * 50)
            print("Start Evaluation!")

            # Evaluate on test data
            hr_A, NDCG_A, hr_B, NDCG_B = self.model.evaluate(
                self.topK, self.testNeg_A[0], self.testNeg_A[1],
                self.testNeg_B[0], self.testNeg_B[1]
            )

            print(
                f"Epoch{epoch} Domain A: {self.dataName_A} TopK: {self.topK} HR: {hr_A}, NDCG: {NDCG_A}")
            print(
                f"Epoch{epoch} Domain B: {self.dataName_B} TopK: {self.topK} HR: {hr_B}, NDCG: {NDCG_B}")
            print("-" * 50)

            # Track metrics
            loss_A_list.append(loss_A)
            loss_B_list.append(loss_B)
            hr_A_list.append(hr_A)
            NDCG_A_list.append(NDCG_A)
            hr_B_list.append(hr_B)
            NDCG_B_list.append(NDCG_B)

            # Update best metrics
            if hr_A > best_hr_A:
                best_hr_A = hr_A
                best_NDCG_A = NDCG_A

            if hr_B > best_hr_B:
                best_hr_B = hr_B
                best_NDCG_B = NDCG_B

        # Print final results
        print("Domain A: Best HR: {}, NDCG: {}; Domain B: Best HR: {}, NDCG: {}"
              .format(best_hr_A, best_NDCG_A, best_hr_B, best_NDCG_B))

        # Prepare results for saving
        bestPerformance = [
            [best_hr_A, best_NDCG_A],
            [best_hr_B, best_NDCG_B]
        ]

        # Save results
        matname = 'E2FEDREC_' + str(self.dataName_A) + '_' + str(
            self.dataName_B) + '_KSize_' + str(self.KSize) + '_Result.mat'
        torch.save({
            'allResults_A': allResults_A,
            'allResults_B': allResults_B,
            'bestPerformance': bestPerformance
        }, matname)

        print("Training complete!")
