# Copyright 2024 Ant Group Co., Ltd.
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

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from secretflow import reveal
from secretflow.device import PYU
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.core.torch import BaseModule, module
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.utils import TorchModel


class CAFEAttack(AttackCallback):
    """

    Attributes:
        attack_party: attack party.
        victim_party: victim party.
        victim_hidden_size: hidden size of victim, excluding batch size
        dummy_fuse_model: dummy fuse model for attacker
        label: labels for attacker to calculate attack metrics
        attack_epoch: epoch list to attack, if you want to attack all epochs, set it to None
        attack_batch: batch list to attack, if you want to attack all batches, set it to None
        epochs: attack epochs for each batch, 100 for cifar10 to get good attack accuracy
        exec_device: device for calculation, 'cpu' or 'cuda'
    """

    def __init__(
        self,
        attack_party: PYU,
        label_party: PYU,
        victim_hidden_size: List[int],
        label_size: List[int] = [10],
        attack_epoch: List[int] = None,
        attack_batch: List[int] = None,
        epochs: int = 100,
        exec_device: str = 'cpu',
        max_iter: int = 80000,
        batch_size: int = 40,
        data_number: int = 800,
        number_of_workers: int = 4,
        learning_rate_first_shot: float = 5e-3,
        learning_rate_double_shot: float = 1e-2,
        cafe_learning_rate: float = 0.01,
        learning_rate_fl: float = 1e-6,
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party

        self.attack_epoch = attack_epoch
        self.attack_batch = attack_batch

        self.victim_hidden_size = victim_hidden_size
        self.label_party = label_party
        self.label_size = label_size
        self.att_epochs = epochs
        self.exec_device = exec_device

        self.metrics = None
        self.attack = True
        self.dummy_middle_output_gradient = None
        self.dummy_middle_input = None
        self.dummy_data = None
        self.dummy_labels = None
        self.real_data = []
        self.real_label = []
        self.victim_base_model_list = []
        self.attacker_fuse_model = None
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.data_number = data_number
        self.number_of_workers = number_of_workers
        self.learning_rate_first_shot = learning_rate_first_shot
        self.learning_rate_double_shot = learning_rate_double_shot
        self.cafe_learning_rate = cafe_learning_rate
        self.learning_rate_fl = learning_rate_fl

    def on_train_batch_begin(self, batch):
        if self.dummy_middle_output_gradient is None:
            dummy_middle_output_gradient = dummy_middle_output_gradient_init(
                number_of_workers=self.number_of_workers,
                data_number=self.data_number,
                feature_space=256,
            )
            self.dummy_middle_output_gradient = dummy_middle_output_gradient
        if self.dummy_middle_input is None:
            self.dummy_middle_input = dummy_middle_input_init(
                number_of_workers=self.number_of_workers,
                data_number=self.data_number,
                feature_space=2048,
            )
        if self.dummy_data is None:
            self.dummy_data, self.dummy_labels = dummy_data_init(
                number_of_workers=self.number_of_workers,
                data_number=self.data_number,
                pretrain=False,
                true_label=None,
            )

    def on_base_forward_begin(self):
        def get_victim_inputs(worker):
            _data_x = worker._data_x.detach().clone()
            return _data_x

        def get_real_label(worker):
            if isinstance(worker.train_y, list):
                label = worker.train_y[0].cpu().numpy()
            else:
                label = worker.train_y.cpu().numpy()
            return label

        def get_victim_model(victim_worker: SLBaseTorchModel):
            return victim_worker.model_base

        def get_attacker_fuse_model(worker):
            return worker.model_fuse

        tmp_model_list = []
        for key in self._workers.keys():
            if key != self.attack_party:
                true_inputs = reveal(self._workers[key].apply(get_victim_inputs))
                base_model = reveal(self._workers[key].apply(get_victim_model))
                self.real_data.append(true_inputs)
                tmp_model_list.append(base_model)

            if key == self.label_party:
                real_label = reveal(self._workers[key].apply(get_real_label))
                self.real_label = real_label
            if key == self.attack_party:
                self.attacker_fuse_model = reveal(
                    self._workers[key].apply(get_attacker_fuse_model)
                )
        self.victim_base_model_list = tmp_model_list

        cafe_attack(
            local_net=self.victim_base_model_list,
            server=self.attacker_fuse_model,
            dummy_data=self.dummy_data,
            dummy_labels=self.dummy_labels,
            real_data=self.real_data,
            real_labels=self.real_label,
            dummy_middle_output_gradient=self.dummy_middle_output_gradient,
            dummy_middle_input=self.dummy_middle_input,
            max_iters=self.max_iter,
            batch_size=self.batch_size,
            data_number=self.data_number,
            number_of_workers=self.number_of_workers,
            learning_rate_first_shot=self.learning_rate_first_shot,
        )


def dummy_middle_output_gradient_init(number_of_workers, data_number, feature_space):
    """
    Initializes the dummy middle output gradient for each worker node.

    Args:
        number_of_workers (int): The number of worker nodes.
        data_number (int): The number of data points for each worker node.
        feature_space (int): The dimension of the feature space.

    Returns:
        list: A list of initialized middle output gradients for each worker node.
    """
    dummy_middle_output_gradient = []
    for _ in range(number_of_workers):
        # Initialize gradient tensor with values uniformly distributed in the range [-8e-4, 8e-4].
        temp_dummy_middle_output_gradient = torch.empty(
            data_number, feature_space
        ).uniform_(
            -8e-4, 8e-4
        )  # very important!
        temp_dummy_middle_output_gradient = torch.nn.Parameter(
            temp_dummy_middle_output_gradient, requires_grad=True
        )
        dummy_middle_output_gradient.append(temp_dummy_middle_output_gradient)
    return dummy_middle_output_gradient


def dummy_middle_input_init(
    number_of_workers, data_number, feature_space, min_val=0, max_val=8e-2, seed=None
):
    """
    Initialize dummy input tensors for multiple workers with optional random seed and value range.

    Args:
        number_of_workers (int): The number of workers for which dummy input will be created.
        data_number (int): The number of data samples for each worker.
        feature_space (int): The dimensionality of the feature space.
        min_val (float, optional): The minimum value for random initialization. Defaults to 0.
        max_val (float, optional): The maximum value for random initialization. Defaults to 8e-2.
        seed (int, optional): A random seed to ensure reproducibility. If None, randomness is not controlled.

    Returns:
        list: A list of PyTorch Parameters where each element is the dummy input tensor for a worker.
    """
    dummy_middle_input = []
    for worker_index in range(number_of_workers):
        if seed is not None:
            torch.manual_seed(seed + worker_index)
        temp_dummy_middle_input = torch.empty(data_number, feature_space).uniform_(
            min_val, max_val
        )
        temp_dummy_middle_input = torch.nn.Parameter(
            temp_dummy_middle_input, requires_grad=True
        )
        dummy_middle_input.append(temp_dummy_middle_input)
    return dummy_middle_input


def dummy_data_init(
    number_of_workers, data_number, pretrain=False, true_label=None, device="cpu"
):
    """
    Initializes dummy data for training or testing purposes.

    Args:
        number_of_workers (int): The number of workers (clients) involved in generating data.
        data_number (int): The number of images generated for each worker.
        pretrain (bool, optional): If True, loads pre-generated data from files. Defaults to False.
        true_label (torch.Tensor, optional): If provided, these labels will be used as output labels.
                                             Defaults to None.
        device (str, optional): The device used for generating random tensors, either "cpu" or "cuda".
                                Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - dummy_images (list of torch.Tensor): A list of generated or loaded dummy images, one per worker.
            - dummy_labels (torch.Tensor): Generated or loaded dummy labels.
    """
    if pretrain:
        dummy_images = []
        for worker_index in range(number_of_workers):
            temp_dummy_image = np.load(f'result/{worker_index}_dummy.npy')
            temp_dummy_image = torch.tensor(temp_dummy_image, dtype=torch.float32)
            dummy_images.append(temp_dummy_image)
        dummy_labels = np.load('result/labels_.npy')
        dummy_labels = torch.tensor(dummy_labels, dtype=torch.float32)
        return dummy_images, dummy_labels

    else:
        dummy_images = []
        for worker_index in range(number_of_workers):
            temp_dummy_image = torch.rand(
                data_number,
                14,
                14,
                dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(worker_index + 1),
            )
            temp_dummy_image = torch.nn.Parameter(temp_dummy_image)
            dummy_images.append(temp_dummy_image)
        if true_label is None:
            dummy_labels = torch.rand(
                data_number,
                10,
                dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(0),
            )
        else:
            dummy_labels = true_label
        dummy_labels = torch.tensor(dummy_labels, dtype=torch.float32)
        dummy_labels = torch.nn.Parameter(dummy_labels)
        return dummy_images, dummy_labels


def cafe_attack(
    local_net,
    server,
    dummy_data,
    dummy_labels,
    max_iters: int = 80000,
    batch_size: int = 40,
    data_number: int = 800,
    real_data: list = None,
    real_labels=None,
    number_of_workers: int = 4,
    dummy_middle_output_gradient=None,
    learning_rate_first_shot: float = 5e-3,
    dummy_middle_input=None,
    learning_rate_double_shot: float = 1e-2,
    cafe_learning_rate: float = 0.01,
    learning_rate_fl: float = 1e-6,
    save_path: str = "./cafe_result",
):
    # print(cafe_attack)
    model_list = [local_net[worker_index] for worker_index in range(number_of_workers)]
    model_list.append(server)
    opt_server = Adam([{"params": server.parameters()}], lr=learning_rate_fl)
    opt_list = []
    for worker_index in range(number_of_workers):
        opt_list.append(
            Adam(
                [{"params": local_net[worker_index].parameters()}], lr=learning_rate_fl
            )
        )
    real_labels = torch.tensor(real_labels)
    real_labels = torch.nn.functional.one_hot(real_labels, num_classes=10).float()
    opt_cafe = OptimizerForCafe(
        number_of_workers=number_of_workers,
        data_number=data_number,
        learning_rate=cafe_learning_rate,
    )
    # OptimizerForCafe
    optimizer2 = OptimizerForMiddleInput(
        number_of_workers, data_number, learning_rate_double_shot, 2048
    )
    for iter in range(max_iters):
        random.seed(iter)
        random_lists = random.sample(list(range(data_number)), batch_size)
        (
            true_gradient,
            batch_real_data,
            real_middle_input,
            _,
            train_loss,
            train_acc,
        ) = take_gradient(
            number_of_workers, random_lists, real_data, real_labels, local_net, server
        )
        print(type(true_gradient))

        server.zero_grad()
        for worker_index in range(number_of_workers):
            local_net[worker_index].zero_grad()

        batch_dummy_middle_output_gradient = take_batch(
            number_of_workers, dummy_middle_output_gradient, random_lists
        )
        optimizer = SGD(
            [{"params": [param]} for param in batch_dummy_middle_output_gradient],
            lr=learning_rate_first_shot,
        )

        for _ in range(10):
            optimizer.zero_grad()
            for worker_index in range(number_of_workers):
                loss = (
                    torch.norm(
                        torch.sum(
                            batch_dummy_middle_output_gradient[worker_index], dim=0
                        )
                        - true_gradient[worker_index + 1][5]
                    )
                    ** 2
                )
                loss.backward(retain_graph=True)

            optimizer.step()

        for worker_index in range(number_of_workers):
            dummy_middle_output_gradient[worker_index].data[random_lists] = (
                batch_dummy_middle_output_gradient[worker_index].detach()
            )

        batch_dummy_middle_input = take_batch(
            number_of_workers, dummy_middle_input, random_lists
        )
        batch_recovered_middle_output_gradient = take_batch(
            number_of_workers, dummy_middle_output_gradient, random_lists
        )

        for e in range(10):
            for worker_index in range(number_of_workers):

                batch_dummy_middle_input[worker_index] = nn.Parameter(
                    batch_dummy_middle_input[worker_index].detach()
                )
            optimizer_middle_inputs = Adam(
                [{"params": [param]} for param in batch_dummy_middle_input],
                lr=learning_rate_double_shot,
            )
            optimizer_middle_inputs.zero_grad()
            g_middle_inputs = [None for _ in range(number_of_workers)]
            for worker_index in range(number_of_workers):
                loss = (
                    torch.norm(
                        true_gradient[worker_index + 1][4].T
                        - torch.matmul(
                            torch.transpose(
                                batch_dummy_middle_input[worker_index], 0, 1
                            ),
                            batch_recovered_middle_output_gradient[worker_index],
                        )
                    )
                    ** 2
                )
                loss.backward(retain_graph=True)
                g_middle_inputs[worker_index] = (
                    batch_dummy_middle_input[worker_index].grad.detach().clone()
                )

            batch_dummy_middle_input = optimizer2.apply_gradients(
                iter,
                batch_size,
                random_lists,
                g_middle_inputs,
                batch_dummy_middle_input,
            )

        for worker_index in range(number_of_workers):
            MSE = nn.MSELoss()(
                real_middle_input[worker_index], batch_dummy_middle_input[worker_index]
            )
            print(
                f'{worker_index} double shot loss:',
                loss.item(),
                'MSE:',
                MSE.item(),
                end="\t",
            )
            dummy_middle_input[worker_index].data[random_lists] = (
                batch_dummy_middle_input[worker_index].detach()
            )
        batch_dummy_data, batch_dummy_label = take_batch_data(
            number_of_workers, dummy_data, dummy_labels, random_lists
        )

        batch_recovered_middle_input = torch.cat(
            take_batch(number_of_workers, dummy_middle_input, random_lists), axis=1
        )

        D, cafe_gradient_x, cafe_gradient_y = cafe_torch(
            number_of_workers,
            batch_dummy_data,
            batch_dummy_label,
            local_net,
            server,
            true_gradient,
            batch_recovered_middle_input,
        )
        batch_dummy_data = opt_cafe.apply_gradients_data(
            iter, random_lists, cafe_gradient_x, batch_dummy_data
        )
        batch_dummy_label = opt_cafe.apply_gradients_label(
            iter, random_lists, cafe_gradient_y, batch_dummy_label
        )
        for worker_index in range(number_of_workers):
            dummy_data[worker_index].data[random_lists] = batch_dummy_data[
                worker_index
            ].detach()
            dummy_labels.data[random_lists] = batch_dummy_label[worker_index].detach()

        psnr = PSNR(batch_real_data, batch_dummy_data)
        # print results
        print(f"iter: {iter}, loss: {train_loss}, test_acc: {train_acc}")

        for c_id in range(len(opt_list)):
            opt_list[c_id].zero_grad()
        opt_server.zero_grad()
        smdata_list = [None for _ in range(number_of_workers)]
        for c_id in range(number_of_workers):
            smdata_list[c_id] = model_list[c_id](batch_real_data[c_id])[1]
        smdata = torch.cat(smdata_list, dim=1)
        outputs = server(smdata)
        loss = nn.CrossEntropyLoss()(outputs, real_labels[random_lists])
        loss.backward()
        # opt_list.step()
        opt_server.step()
        for c_id in range(len(opt_list)):
            opt_list[c_id].step()

        if iter % 100 == 0:
            # test accuracy
            loss, test_acc = test(
                number_of_workers, real_data, real_labels, local_net, server
            )
            print(
                f"D: {D}, psnr: {psnr}, iter: {iter}, train_loss: {train_loss}, test_acc: {test_acc}"
            )

    visual_data(real_data, True, save_path)
    visual_data(dummy_data, False, save_path)


class OptimizerForMiddleInput:
    def __init__(
        self,
        number_of_workers,
        data_number,
        learning_rate,
        feature_space=2048,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.h_data = []
        self.v_data = []
        self.number_of_workers = number_of_workers

        for worker_index in range(number_of_workers):
            self.h_data.append(torch.zeros([data_number, feature_space]))
            self.v_data.append(torch.zeros([data_number, feature_space]))

    def apply_gradients(self, iter, batchsize, random_lists, gradient, theta):
        """
        Applies the gradients to update the model parameters (theta) for each worker (client) using a variant of
        the Adam optimization algorithm.

        Args:
            iter (int): The current iteration step in the optimization process.
            batchsize (int): The size of the batch used for this iteration.
            random_lists (Tensor): A list of indices randomly selected for this batch.
            gradient (List[Tensor]): A list of gradients from different workers to be applied.
            theta (List[Tensor]): A list of model parameters (weights) for each worker to be updated.

        Returns:
            List[Tensor]: A list of updated model parameters (theta) for each worker.

        Updates the `h_data` and `v_data` attributes for each worker using the provided gradients.
        These attributes store the first and second moments (h and v) of the gradients, respectively.
        """
        theta_new = []
        temp_lr = (
            self.lr
            * math.sqrt(1 - self.beta2 ** (iter + 1))
            / (1 - self.beta1 ** (iter + 1))
        )

        for worker_index in range(self.number_of_workers):
            h = self.h_data[worker_index][random_lists]
            h = self.beta1 * h + (1 - self.beta1) * gradient[worker_index]
            v = self.v_data[worker_index][random_lists]
            v = self.beta2 * v + (1 - self.beta2) * torch.square(gradient[worker_index])
            h_hat = h / (1 - self.beta1 ** (iter + 1))
            v_hat = v / (1 - self.beta2 ** (iter + 1))
            for batch_index in range(batchsize):
                self.h_data[worker_index][random_lists[batch_index], :] = h[
                    batch_index, :
                ]
                self.v_data[worker_index][random_lists[batch_index], :] = v[
                    batch_index, :
                ]
            temp_theta = theta[worker_index] - temp_lr * h_hat / (
                torch.sqrt(v_hat) + self.epsilon
            )
            theta_new.append(temp_theta)

        return theta_new


class OptimizerForCafe:
    """
    Class for optimizing parameters in a cafe-related optimization problem.

    Args:
        number_of_workers: Number of workers.
        data_number: Number of data points.
        learning_rate: Learning rate for optimization.
        beta1 (optional): Default value is 0.9.
        beta2 (optional): Default value is 0.999.
        epsilon (optional): Default value is 1e-7.
    """

    def __init__(
        self,
        number_of_workers,
        data_number,
        learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.number_of_workers = number_of_workers
        self.h_data = []
        self.v_data = []

        for worker_index in range(number_of_workers):
            self.h_data.append(torch.zeros([data_number, 14, 14]))
            self.v_data.append(torch.zeros([data_number, 14, 14]))

        self.h_label = torch.zeros([data_number, 10])
        self.v_label = torch.zeros([data_number, 10])

    def apply_gradients_data(self, iter, random_lists, gradient, theta):
        """
        Applies gradients to data for optimization.

        Args:
            iter: Iteration number.
            random_lists: Random lists for indexing.
            gradient: Gradient tensors.
            theta: Current parameters.

        Returns:
            Updated parameters.
        """
        theta_new = []
        temp_lr = (
            self.lr
            * math.sqrt(1 - self.beta2 ** (iter + 1))
            / (1 - self.beta1 ** (iter + 1))
        )

        for worker_index in range(self.number_of_workers):
            h = self.h_data[worker_index][random_lists]
            h = self.beta1 * h + (1 - self.beta1) * gradient[worker_index]
            v = self.v_data[worker_index][random_lists]
            v = self.beta2 * v + (1 - self.beta2) * torch.square(gradient[worker_index])
            h_hat = h / (1 - self.beta1 ** (iter + 1))
            v_hat = v / (1 - self.beta2 ** (iter + 1))
            temp_theta = theta[worker_index] - temp_lr * h_hat / (
                torch.sqrt(v_hat) + self.epsilon
            )
            theta_new.append(temp_theta)
            for batch_index in range(len(random_lists)):
                self.h_data[worker_index][random_lists[batch_index], :, :] = h[
                    batch_index, :, :
                ]
                self.v_data[worker_index][random_lists[batch_index], :, :] = v[
                    batch_index, :, :
                ]

        return theta_new

    def apply_gradients_label(self, iter, random_lists, gradient, theta):
        temp_lr = (
            self.lr
            * math.sqrt(1 - self.beta2 ** (iter + 1))
            / (1 - self.beta1 ** (iter + 1))
        )
        h = self.h_label[random_lists]
        h = self.beta1 * h + (1 - self.beta1) * gradient
        v = self.v_label[random_lists]
        v = self.beta2 * v + (1 - self.beta2) * torch.square(gradient)
        h_hat = h / (1 - self.beta1 ** (iter + 1))
        v_hat = v / (1 - self.beta2 ** (iter + 1))
        theta_new = theta - temp_lr * h_hat / (torch.sqrt(v_hat) + self.epsilon)
        for batch_index in range(len(random_lists)):
            self.h_label[random_lists[batch_index], :] = h[batch_index, :]
            self.v_label[random_lists[batch_index], :] = v[batch_index, :]

        return theta_new


def visual_data(data, real, save_path):
    """
    This function is used to visualize data.

    Args:
        data (list): The data to be visualized.
        real (bool): A flag indicating whether the data is real.
        save_path (str): The path where the visualized images will be saved.

    Returns:
        None
    """
    import os

    import matplotlib.pyplot as plt

    def create_path_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    number_of_worker = len(data)
    if real:
        for worker_index in range(number_of_worker):
            data_number = data[worker_index].shape[0]
            for data_index in range(data_number):
                data_to_be_visualized = data[worker_index][data_index, :, :].numpy()
                data_to_be_visualized = data_to_be_visualized.reshape([14, 14])
                plt.imshow(data_to_be_visualized)
                create_path_if_not_exists(f'{save_path}/{worker_index}')
                plt.savefig(f'{save_path}/{worker_index}/{data_index}real.png')
                plt.close()
    else:
        for worker_index in range(number_of_worker):
            data_number = data[worker_index].shape[0]
            for data_index in range(data_number):
                data_to_be_visualized = (
                    data[worker_index][data_index, :, :].cpu().detach().numpy()
                )
                data_to_be_visualized = data_to_be_visualized.reshape([14, 14])
                plt.imshow(data_to_be_visualized)
                create_path_if_not_exists(f'{save_path}/{worker_index}')
                plt.savefig(f'{save_path}/{worker_index}/{data_index}dummy.png')
                plt.close()


def compute_loss(labels, logits):
    """
    Computes the cross-entropy loss.

    Args:
        labels (Tensor): The indices of the target classes.
        logits (Tensor): The raw network output, i.e., unnormalized predictions before applying softmax.

    Returns:
        Tensor: The computed loss value.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(logits, labels)


def compute_accuracy(labels, logits):
    """
    Computes classification accuracy.
    Args:
        labels: Indices of target classes.
        logits: Raw outputs of the network without softmax.
    Returns:
        Computed accuracy value.
    """
    predicted = logits.max(dim=-1)[-1]  # 获取每个样本的预测类别
    correct = (predicted == labels.max(dim=-1)[-1]).sum().item()  # 计算正确预测的数量
    return correct / len(labels)  # 计算准确率


def total_variation(x):
    """
    Computes the total variation (TV) of an image.
    Args:
        x: Input image with shape [N, C, H, W].
    Returns:
        Computed total variation value.
    """
    # Calculate horizontal variation.
    tv_h = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    # Calculate vertical variation.
    tv_w = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    # Return the sum of total variations.
    return tv_h + tv_w


def take_gradient(
    number_of_workers,
    random_lists,
    real_data,
    real_labels,
    local_net,
    server,
    fake_grad=False,
):
    """
    Computes gradients for the server and local models.

    Args:
        number_of_workers (int): The number of workers.
        random_lists (list): Random lists for data indexing.
        real_data (list): Real data for training.
        real_labels (torch.Tensor): Real labels for training.
        local_net (list): List of local neural networks.
        server (Server): The server model.
        fake_grad (bool, optional): Whether to use fake gradients. Defaults to False.

    Returns:
        tuple: A tuple containing gradients and various intermediate values.
    """
    true_gradient = []
    local_output = []
    middle_input = []
    middle_output = []
    batch_real_data = []
    real_tv_norm = []

    with torch.autograd.set_grad_enabled(True):
        # Select data for the current worker and convert to appropriate dtype
        labels = real_labels[random_lists]
        labels = labels
        for worker_index in range(number_of_workers):
            temp_data = real_data[worker_index][random_lists]
            temp_data = temp_data.to(dtype=torch.float32)
            temp_data = temp_data
            # Mark the data as requiring gradient computation
            temp_data.requires_grad_()

            # Pass the data through the local network and store intermediate outputs
            temp_middle_input, temp_local_output, temp_middle_output = local_net[
                worker_index
            ](temp_data)
            middle_input.append(temp_middle_input)
            temp_middle_output.retain_grad()  # 保留梯度
            middle_output.append(temp_middle_output)
            local_output.append(temp_local_output)
            batch_real_data.append(temp_data)

            # Reshape data and compute total variation norm
            temp_data = temp_data.view(-1, 1, 14, 14)
            temp_tv_norm = total_variation(temp_data)
            real_tv_norm.append(temp_tv_norm)

        # Concatenate middle inputs and local outputs
        real_local_output = torch.cat(local_output, dim=1)

        # Pass the local outputs through the server model to get predictions
        predict = server(real_local_output)

        loss = compute_loss(labels, predict)
        train_acc = compute_accuracy(labels, predict)
        server.zero_grad()

        for worker_index in range(number_of_workers):
            local_net[worker_index].zero_grad()

        loss.backward()
        server_true_gradient = [p.grad for p in server.parameters()]
        true_gradient.append(server_true_gradient)

        # Compute gradients for local models
        middle_output_gradient = []
        for worker_index in range(number_of_workers):
            local_true_gradient = [p.grad for p in local_net[worker_index].parameters()]
            if fake_grad:
                # local_true_gradient = _fake_grad(local_true_gradient)
                pass
            middle_output_gradient.append([o.grad for o in middle_output[worker_index]])
            true_gradient.append(local_true_gradient)

    return (
        true_gradient,
        batch_real_data,
        middle_input,
        middle_output_gradient,
        loss,
        train_acc,
    )


def take_batch(number_of_workers, dummy_item, random_lists):
    batch_dummy_item = []
    for worker_index in range(number_of_workers):
        temp_dummy_item = dummy_item[worker_index][random_lists]
        temp_dummy_item = torch.nn.Parameter(temp_dummy_item)
        batch_dummy_item.append(temp_dummy_item)
    return batch_dummy_item


def take_batch_data(number_of_workers, dummy_images, dummy_labels, random_lists):
    batch_dummy_images = []
    for worker_index in range(number_of_workers):
        temp_dummy_image = dummy_images[worker_index][random_lists]
        temp_dummy_image = nn.Parameter(temp_dummy_image)
        batch_dummy_images.append(dummy_images[worker_index][random_lists])
    batch_dummy_labels = nn.Parameter(dummy_labels[random_lists])
    return batch_dummy_images, batch_dummy_labels


def cafe_torch(
    number_of_workers,
    batch_dummy_image,
    batch_dummy_label,
    local_net,
    server,
    real_gradient,
    real_middle_input,
):
    '''
    Core part of the algorithm: DLG.
    :param number_of_workers: The number of workers.
    :param batch_dummy_image: A batch of dummy images.
    :param batch_dummy_label: A batch of dummy labels.
    :param local_net: The local network.
    :param server: The server.
    :param real_gradient: The real gradient.
    :return: D, dlg_gradient_x, dlg_gradient_y.
    '''
    # Initialize lists to store fake gradients and outputs
    fake_gradient = []
    fake_local_output = []
    fake_middle_input = []
    batch_dummy_label = nn.Parameter(batch_dummy_label)

    for worker_index in range(number_of_workers):
        batch_dummy_image[worker_index] = nn.Parameter(batch_dummy_image[worker_index])
        temp_middle_input, temp_local_output, temp_middle_output = local_net[
            worker_index
        ](batch_dummy_image[worker_index])
        fake_local_output.append(temp_local_output)
        fake_middle_input.append(temp_middle_input)

    dummy_middle_input = torch.cat(fake_middle_input, dim=1)
    dummy_local_output = torch.cat(fake_local_output, dim=1)

    predict = server(dummy_local_output)
    true = F.softmax(batch_dummy_label, dim=1)
    loss = compute_loss(true, predict)
    server.zero_grad()
    for worker_index in range(number_of_workers):
        local_net[worker_index].zero_grad()
    loss.backward(retain_graph=True)
    temp_server_true_gradient = [param.grad for param in server.parameters()]
    fake_gradient.append(temp_server_true_gradient)

    for worker_index in range(number_of_workers):
        temp_local_fake_gradient = [
            param.grad for param in local_net[worker_index].parameters()
        ]
        fake_gradient.append(temp_local_fake_gradient)

    D = 0
    for layer in range(len(real_gradient)):
        for gr, gf in zip(real_gradient[layer], fake_gradient[layer]):
            gr = gr.view(-1, 1)
            gf = gf.view(-1, 1)
            D += torch.norm(gr - gf) ** 2
    D *= 100

    D_local_output_norm = 0
    for r_real_middle_input, dummy_middle_input_item in zip(
        real_middle_input, dummy_middle_input
    ):
        temp_input_norm = torch.norm(r_real_middle_input - dummy_middle_input_item) ** 2
        D_local_output_norm += temp_input_norm

    print("CAFE loss: %.5f" % D.item(), end='\t')
    print('Input norm:', D_local_output_norm.item(), end='\t')

    tv_norm = []
    for worker_index in range(number_of_workers):
        temp_data = batch_dummy_image[worker_index]
        temp_data = temp_data.view(-1, 14, 14, 1)
        temp_tv_norm = total_variation(temp_data)
        tv_norm.append(temp_tv_norm)

    tv_norm_aggregated = tv_norm[0]
    for worker_index in range(1, number_of_workers):
        tv_norm_aggregated += tv_norm[worker_index]
    tv_norm_aggregated = tv_norm_aggregated / number_of_workers
    tv_norm_aggregated = tv_norm_aggregated.mean()

    print('with Tv norm', tv_norm_aggregated.item(), end='\t')

    cafe_gradient_x = []
    opt_batch_dummy_label = Adam([batch_dummy_label], lr=0.0001)
    opt_batch_dummy_label.zero_grad()
    opt_batch_dummy_label = Adam(
        [{"params": [param]} for param in batch_dummy_image], lr=0.0001
    )
    opt_batch_dummy_label.zero_grad()
    loss.backward(retain_graph=True)
    cafe_gradient_y = batch_dummy_label.grad.clone()

    for worker_index in range(number_of_workers):
        batch_dummy_image[worker_index].grad = None
        loss.backward(retain_graph=True)
        temp_local_output_gradient = batch_dummy_image[worker_index].grad.clone()

        batch_dummy_image[worker_index].grad = None
        D_local_output_norm.backward(retain_graph=True)
        temp_dlg_gradient = 1e-4 * batch_dummy_image[worker_index].grad.clone()

        temp_cafe_gradient_x = 1e-3 * temp_local_output_gradient + temp_dlg_gradient
        if tv_norm_aggregated.item() > 25:
            batch_dummy_image[worker_index].grad = None
            tv_norm[worker_index].backward(retain_graph=True)
            temp_tv_norm_gradient = batch_dummy_image[worker_index].grad.clone()
            temp_cafe_gradient_x = temp_cafe_gradient_x + 1e-4 * temp_tv_norm_gradient

        cafe_gradient_x.append(temp_cafe_gradient_x)

    return D.item(), cafe_gradient_x, cafe_gradient_y


def PSNR(batch_real_image, batch_dummy_image):
    '''
    Compute PSNR
    :param batch_real_image:
    :param batch_dummy_image:
    :return: PSNR value
    '''
    psnr = []
    for worker_index in range(len(batch_real_image)):
        dummy = torch.clamp(batch_dummy_image[worker_index], 0, 1).view(-1, 1, 14, 14)
        real = batch_real_image[worker_index].view(-1, 1, 14, 14)
        mse = torch.mean((real - dummy) ** 2)
        psnr_value = 10 * torch.log10(1 / mse)
        psnr.append(psnr_value)

    aggregated_psnr = torch.mean(torch.stack(psnr))
    print('psnr value:', aggregated_psnr.item(), end='\t')
    return aggregated_psnr.item()


def test(number_of_workers, test_data, test_labels, local_net, server):
    local_output = []
    test_labels = test_labels
    for worker_index in range(number_of_workers):
        # compute output
        test_data[worker_index] = test_data[worker_index].float()
        # print(test_data[worker_index])
        temp_middle_input, temp_local_output, temp_middle_output = local_net[
            worker_index
        ](test_data[worker_index])
        # collect terms
        local_output.append(temp_local_output)
    # concatenate
    real_local_output = torch.cat(local_output, axis=1)  # batch size x 40
    # server part
    predict = server(real_local_output)
    # compute loss
    loss = compute_loss(test_labels, predict)
    # training accuracy
    test_acc = compute_accuracy(test_labels, predict)
    print(f"loss: {loss}, test_acc: {test_acc}")
    return loss, test_acc
