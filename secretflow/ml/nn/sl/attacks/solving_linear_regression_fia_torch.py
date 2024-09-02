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

import logging

import numpy as np
import torch

from secretflow import reveal
from secretflow.device import PYU
from secretflow.ml.nn.callbacks.attack import AttackCallback


class SolvingLinearRegressionAttack(AttackCallback):
    """
    the feature reconstruction attack of DNN training in Vertical Federated Learning.
    Reference: https://arxiv.org/pdf/2210.06771.

    Attributes:
        attack_party (PYU): The party performing the attack.
        victim_party (PYU): The party being attacked.
        targets_columns (list): List of target column indices, default is [4, 5, 6].
        r (int): The number of rows to sample and rescale using leverage score sampling, defaults to 9.
        exec_device (str): Device used for computation, either 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        attack_party: PYU,
        victim_party: PYU,
        targets_columns: list = [4, 5, 6],
        r: int = 9,
        exec_device: str = 'cpu',
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.victim_party = victim_party

        # for attacker
        self.victim_model_dict = {}
        self.targets_columns = targets_columns
        self.r = r
        self.logs = {}
        self.exec_device = exec_device

        self.metrics = {}

    def on_epoch_begin(self, epoch=None, logs=None):
        """
        This method is called at the beginning of each epoch.

        It defines two helper functions:
        - reset_truth_feature: Resets the 'feature' in the callback store of the victim party's worker to an empty array.
        - reset_predict_feature: Resets the 'predict_feature' in the callback store of the attack party's worker to an empty array.

        Then, it applies the reset_truth_feature function to the workers of the victim party and the reset_predict_feature function to the workers of the attack party.
        """

        def reset_truth_feature(worker):
            worker._callback_store['solving_attack'] = {
                'feature': np.empty((0, len(self.targets_columns)))
            }

        def reset_predict_feature(worker):
            worker._callback_store['solving_attack'] = {
                'predict_feature': np.empty((0,))
            }

        self._workers[self.victim_party].apply(reset_truth_feature)
        self._workers[self.attack_party].apply(reset_predict_feature)

    def on_fuse_forward_begin(self):
        """
        This method is executed at the beginning of the fusion forward process.

        It defines several helper functions to handle data from workers and perform attack-related computations.

        The functions defined are:
        - get_victim_hidden: Retrieves the hidden state from the victim worker.
        - get_victim_inputs: Appends the input data corresponding to the target columns from the victim worker.
        - solving_attack: Performs a linear regression-based attack using the hidden state.

        Then, it retrieves the hidden output from the victim party, applies the input-related function to the victim party, and applies the attack function to the attack party using the retrieved hidden output.

        Finally, it calls the super class's on_fuse_forward_begin method.
        """

        def get_victim_hidden(worker):
            h = worker._h.detach()
            return h

        def get_victim_inputs(worker):
            _data_x = worker._data_x.detach().clone()
            assert self.targets_columns is not None
            worker._callback_store['solving_attack']['feature'] = np.append(
                worker._callback_store['solving_attack']['feature'],
                _data_x[:, self.targets_columns],
                axis=0,
            )

        def solving_attack(worker, h):
            U, S, V = torch.svd(h.detach().cpu())
            d = (S > 1e-10).sum().item()
            indices = np.arange(d)
            A = h.detach().cpu()[:, indices]
            x_star = self.solve_linear_regression(A, min(self.r, d))
            # print("x_star", x_star)
            worker._callback_store['solving_attack']['predict_feature'] = np.append(
                worker._callback_store['solving_attack']['predict_feature'],
                x_star,
                axis=0,
            )

        hidden_output = reveal(
            self._workers[self.victim_party].apply(get_victim_hidden)
        )
        self._workers[self.victim_party].apply(get_victim_inputs)
        self._workers[self.attack_party].apply(solving_attack, h=hidden_output)

        return super().on_fuse_forward_begin()

    def on_epoch_end(self, epoch=None, logs=None):
        """
        This method is called at the end of each epoch.

        It retrieves the true feature and predicted feature from the workers and computes the attack accuracy.

        Args:
        epoch (int): The current epoch number.
        logs (dict): Logging information.

        Steps:
        1. Retrieve the true feature from the victim party's worker.
        2. Retrieve the predicted feature from the attack party's worker.
        3. For each column, compare the predicted and true features to count the correct matches.
        4. Calculate the accuracy for each column and store it in a list.
        5. Update the "Attack_Accuracy" metric with the maximum accuracy from the list.

        Returns:
        The result of the super class's on_epoch_end method.
        """
        true_feature = reveal(
            self._workers[self.victim_party].apply(
                lambda worker: worker._callback_store['solving_attack']['feature']
            )
        )

        predict_feature = reveal(
            self._workers[self.attack_party].apply(
                lambda worker: worker._callback_store['solving_attack'][
                    'predict_feature'
                ]
            )
        )
        res = []
        for column in range(len(self.targets_columns)):
            # 比较每一列 找出最好的一列
            matching_elements = np.abs(predict_feature - true_feature[:, column]) < 1e-5
            correct_count = np.sum(matching_elements)
            res.append(correct_count / len(predict_feature))
        self.metrics["Attack_Accuracy"] = max(res)

        return super().on_epoch_end(epoch, logs)

    def get_attack_metrics(self):
        return self.metrics

    @staticmethod
    def solve_linear_regression(A, r):
        """
        This function solves the linear regression problem using leverage score sampling.

        Args:
        A (torch.Tensor): The input matrix for the linear regression.
        r (int): The number of rows to sample in the leverage score sampling.

        Steps:
        1. Get the sampled matrix A_prime, sampled indices, and diagonal matrix D using the leverage_score_sampling function.
        2. Initialize a list T with a zero vector where the first element is 1.
        3. Generate different binary vectors x_prime and calculate corresponding w_prime using the sampled matrix.
        4. Create the n-dimensional vector x based on the sampled indices and predictions.
        5. Append each x to the list T.
        6. Find the x_star that minimizes the loss among all x in T.

        Returns:
        x_star (torch.Tensor): The optimal x that minimizes the loss.
        """
        A_prime, sampled_indices, D = leverage_score_sampling(A, r)

        T = [torch.zeros(A.size(0))]
        T[0][0] = 1.0  # e = [1, 0, ..., 0] in R^n
        for i in range(1, 2**r):
            x_prime = torch.tensor(
                list(map(int, bin(i)[2:].zfill(r))), dtype=torch.float32
            )
            result = torch.linalg.lstsq(A_prime, (D @ x_prime).unsqueeze(1))
            w_prime = result.solution.squeeze(1)

            # Create n-dimensional vector x
            x = torch.zeros(A.size(0))
            for idx, x_prime_value in zip(sampled_indices, x_prime):
                x[idx] = x_prime_value

            for j in range(A.size(0)):
                if j in sampled_indices:
                    x[j] = x[j]
                elif (A @ w_prime)[j] < 0.5:
                    x[j] = 0.0
                else:
                    x[j] = 1.0

            T.append(x)

        x_star = None
        min_loss = float("inf")
        for x in T[:]:
            # Solve for w using least squares for the current x
            result = torch.linalg.lstsq(A, x.unsqueeze(1))
            # result = least_squares_solution(A, x.unsqueeze(1))
            w = result.solution.squeeze(1)

            # Calculate the loss ||A @ w - x||^2
            _loss = torch.norm(A @ w - x, p=2).item()
            if _loss < min_loss:
                min_loss = _loss
                x_star = x
        return x_star


def leverage_score_sampling(A, r):
    """
    This function performs leverage score sampling on the input matrix A.

    Args:
    A (torch.Tensor): The input matrix.
    r (int): The number of rows to sample.

    Steps:
    1. Calculate the leverage scores by performing SVD on A and summing the squared rows of U.
    2. Sample r rows based on the calculated leverage scores, using the probabilities derived from the scores.
    3. Construct the sampling matrix S, a zero matrix with 1s at the sampled row indices, and the diagonal matrix D with appropriate values.
    4. Compute the sampled matrix A_prime by multiplying D, S, and A.

    Returns:
    A_prime (torch.Tensor): The sampled matrix.
    sampled_indices (numpy.ndarray): The indices of the sampled rows.
    D (torch.Tensor): The diagonal matrix.
    """
    # Step 1: Calculate leverage scores
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    leverage_scores = torch.sum(U**2, dim=1)

    # Step 2: Sample r rows based on leverage scores
    probabilities = leverage_scores / leverage_scores.sum()
    sampled_indices = np.random.choice(
        A.size(0), size=r, replace=False, p=probabilities.numpy()
    )

    # Step 3: Construct the sampling matrix S and diagonal matrix D
    S = torch.zeros(r, A.size(0))
    for i, idx in enumerate(sampled_indices):
        S[i, idx] = 1.0

    D = torch.diag(1.0 / torch.sqrt(r * probabilities[sampled_indices]))

    A_prime = D @ S @ A

    return A_prime, sampled_indices, D


def select_rank_d_submatrix(Z_A, d):
    """
    This function selects a rank-d submatrix from the input matrix Z_A.

    Args:
    Z_A (numpy.ndarray): The input matrix.
    d (int): The desired rank of the submatrix.

    Returns:
    A (numpy.ndarray): The selected rank-d submatrix.
    selected_indices (numpy.ndarray): The indices of the columns selected to form the submatrix.

    Raises:
    ValueError: If d is greater than the number of columns in Z_A or if a rank-d submatrix cannot be found.
    """
    n, d_A = Z_A.shape
    if d > d_A:
        raise ValueError("d must be less than or equal to d_A")

    # 选择列的索引
    indices = np.arange(d_A)
    np.random.shuffle(indices)  # 随机打乱列的索引

    for i in range(d_A - d + 1):
        selected_indices = indices[i : i + d]
        A = Z_A[:, selected_indices]

        # 检查选出的子矩阵的秩是否为d
        if np.linalg.matrix_rank(A) == d:
            return A, selected_indices

    raise ValueError("Unable to find a rank-d submatrix with the given columns")
