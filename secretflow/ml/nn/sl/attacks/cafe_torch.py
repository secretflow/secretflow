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
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from secretflow import reveal
from secretflow.device import PYU
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel


class CAFEAttack(AttackCallback):
    """
    CAFE attack class for Split Learning (SL) scenarios, enabling the attacker to reconstruct intermediate
    features from victim models using gradient information.

    Args:
        attack_party (PYU): The attacking party.
        label_party (PYU): The label party responsible for providing true labels during the attack.
        victim_hidden_size (List[int]): The hidden size of the victim model, excluding the batch size.
        lr (float): The learning rate for the attack. Default is 0.05.
        real_data_for_save (list): A list of real data to save for analysis. Default is None.
        label_size (List[int]): The size of the label vector. Default is [10].
        attack_epoch (List[int]): A list of epochs to attack. Set to None to attack all epochs. Default is None.
        attack_batch (List[int]): A list of batches to attack. Set to None to attack all batches. Default is None.
        exec_device (str): The device used for executing the attack, either 'cpu' or 'cuda'. Default is 'cpu'.
        batch_size (int): The batch size used during the attack. Default is 40.
        data_number (int): The total number of data samples used in the attack. Default is 800.
        number_of_workers (int): The number of workers used for data loading. Default is 4.
        learning_rate_first_shot (float): The learning rate for the first stage of the attack. Default is 5e-3.
        learning_rate_double_shot (float): The learning rate for the second stage of the attack. Default is 1e-2.
        cafe_learning_rate (float): The learning rate for the CAFE attack method. Default is 0.01.
        learning_rate_fl (float): The learning rate for federated learning scenarios. Default is 1e-6.
        save_image: bool = True,
        **params: Additional parameters for customization.
    """

    def __init__(
        self,
        attack_party: PYU,
        label_party: PYU,
        victim_hidden_size: List[int],
        lr: float = 0.05,
        real_data_for_save: list = None,
        label_size: List[int] = [10],
        attack_epoch: List[int] = None,
        attack_batch: List[int] = None,
        exec_device: str = 'cpu',
        batch_size: int = 40,
        data_number: int = 800,
        number_of_workers: int = 4,
        learning_rate_first_shot: float = 5e-3,
        learning_rate_double_shot: float = 1e-2,
        cafe_learning_rate: float = 0.01,
        learning_rate_fl: float = 1e-6,
        save_image: bool = True,
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
        self.lr = lr
        self.label_size = label_size
        self.exec_device = exec_device
        self.real_data_for_save = real_data_for_save
        self.logs = {}
        self.metrics = None
        self.attack = True
        self.true_grad_list = []
        self.dummy_middle_output_gradient = None
        self.dummy_middle_input = None
        self.dummy_data = None
        self.dummy_labels = None
        self.real_data = []
        self.real_label = []
        self.victim_base_model_list = []
        self.attacker_fuse_model = None
        self.batch_size = batch_size
        self.data_number = data_number
        self.number_of_workers = number_of_workers
        self.learning_rate_first_shot = learning_rate_first_shot
        self.learning_rate_double_shot = learning_rate_double_shot
        self.cafe_learning_rate = cafe_learning_rate
        self.learning_rate_fl = learning_rate_fl

        self.true_gradient = []
        self.clients_outputs = []
        self.iter_num = -1
        self.save_image = save_image
        self.metrics = None

    def on_train_begin(self, logs=None):
        """
        Initializes the dummy gradients, inputs, and data used in the attack.

        Args:
            logs (dict, optional): Logs of the training process. Defaults to None.
        """
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

    def on_train_batch_begin(self, batch):
        self.dummy_index = list(
            range(batch * self.batch_size, self.batch_size * (batch + 1))
        )
        self.iter_num += 1

        return super().on_train_batch_begin(batch)

    def on_fuse_forward_begin(self):
        self.clients_outputs = []

        def get_client_outputs(worker):
            return worker._h.detach()

        for key in self._workers.keys():
            if key != self.attack_party:
                real_h = reveal(self._workers[key].apply(get_client_outputs))
                self.clients_outputs.append(real_h)
        self.true_gradient = []

        self.real_data = []

        def get_attacker_gradient(worker, clients_outputs):
            worker._callback_store['cafe_attack'] = {}
            worker._callback_store['cafe_attack']['true_gradient'] = []
            h = torch.cat(clients_outputs, dim=1)

            outputs = worker.model_fuse(h)
            grad_outputs = [torch.ones_like(outputs)]
            real_grad = torch.autograd.grad(
                outputs=outputs,
                inputs=worker.model_fuse.parameters(),
                grad_outputs=grad_outputs,
            )
            for param in worker.model_fuse.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            worker._callback_store['cafe_attack']['true_gradient'] = [list(real_grad)]
            return list(real_grad)

        real_grad = reveal(
            self._workers[self.attack_party].apply(
                get_attacker_gradient, self.clients_outputs
            )
        )
        self.true_gradient.append(real_grad)

    def on_base_backward_begin(self):

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

        def check_fake_gradient(worker):
            return (
                len(worker._callback_store['cafe_attack']['true_gradient'])
                == self.number_of_workers + 1
            )

        def get_client_upload_grad(worker):
            return worker._callback_store['cafe_attack']['true_gradient']

        def get_gradient(worker):
            real_grad = torch.autograd.grad(
                worker._h,
                worker.model_base.parameters(),
                grad_outputs=worker._gradient,
                retain_graph=True,
            )
            return real_grad

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

        check_fake_grad = reveal(
            self._workers[self.attack_party].apply(check_fake_gradient)
        )

        if not check_fake_grad:
            for key in self._workers.keys():
                if key != self.attack_party:
                    real_grad = reveal(self._workers[key].apply(get_gradient))
                    real_grad = list(real_grad)
                    self.true_gradient.append(real_grad)
        else:
            self.true_gradient = reveal(
                self._workers[self.attack_party].apply(get_client_upload_grad)
            )

        (
            self.dummy_data,
            self.dummy_labels,
            self.dummy_middle_output_gradient,
            self.dummy_middle_input,
        ) = cafe_attack(
            local_net=self.victim_base_model_list,
            server=self.attacker_fuse_model,
            dummy_data=self.dummy_data,
            dummy_labels=self.dummy_labels,
            real_data=self.real_data,
            real_labels=self.real_label,
            dummy_middle_output_gradient=self.dummy_middle_output_gradient,
            dummy_middle_input=self.dummy_middle_input,
            batch_size=self.batch_size,
            data_number=self.data_number,
            number_of_workers=self.number_of_workers,
            learning_rate_first_shot=self.learning_rate_first_shot,
            true_gradient_list=self.true_gradient,
            dummy_index=self.dummy_index,
            iter_num=self.iter_num,
        )

    def on_train_end(self, logs=None):
        if self.save_image:
            visual_data(self.real_data_for_save, True)
            visual_data(self.dummy_data, False)
        self.metrics = PSNR(self.real_data_for_save, self.dummy_data)

    def get_attack_metrics(self):
        return self.metrics


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
    for worker_index in range(number_of_workers):
        temp_dummy_middle_output_gradient = torch.empty(
            data_number, feature_space
        ).uniform_(-8e-4, 8e-4)
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


def visual_data(data, real, save_path="./result/"):
    """
    This function is used to visualize data.

    Args:
        data (list): The data to be visualized.
        real (bool): A flag indicating whether the data is real.
        save_path (str): The path where the visualized images will be saved.

    Returns:
        None
    """

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


def take_batch(number_of_workers, dummy_item, random_lists):
    """
    Selects a batch of dummy items for each worker and converts them to PyTorch parameters.

    Args:
        number_of_workers (int): The number of workers involved in the operation.
        dummy_item (list): A list of dummy items, where each element corresponds to a worker's data.
        random_lists (list or Tensor): A list or tensor of indices used to randomly select items for each worker.

    Returns:
        list: A list containing PyTorch `Parameter` objects, with one batch of dummy items for each worker.

    Example:
        batch = take_batch(4, dummy_item, random_lists)
        This will return a list of parameters, where each parameter corresponds to the selected dummy items for a worker.
    """
    batch_dummy_item = []
    for worker_index in range(number_of_workers):
        temp_dummy_item = dummy_item[worker_index][random_lists]
        temp_dummy_item = torch.nn.Parameter(temp_dummy_item)
        batch_dummy_item.append(temp_dummy_item)
    return batch_dummy_item


def take_batch_data(number_of_workers, dummy_images, dummy_labels, random_lists):
    """
    Selects a batch of dummy images and labels for each worker, and converts the selected items into PyTorch parameters.

    Args:
        number_of_workers (int): The number of workers involved in the operation.
        dummy_images (list): A list of dummy images, where each element corresponds to the data for a specific worker.
        dummy_labels (Tensor): A tensor of dummy labels shared across workers.
        random_lists (list): A list of indices used to randomly select a subset of images and labels.

    Returns:
        tuple:
            - batch_dummy_images (list): A list of selected dummy images for each worker, where each batch is a PyTorch `Parameter`.
            - batch_dummy_labels (Parameter): A PyTorch `Parameter` containing the selected dummy labels.

    Example:
        batch_images, batch_labels = take_batch_data(4, dummy_images, dummy_labels, random_lists)
        This will return a list of batch images for each worker and a batch of labels for the selected indices.
    """
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
    """
    Executes the core part of the CAFE attack algorithm, implementing Deep Leakage from Gradients (DLG).

    This function computes the gradient differences between real and fake gradients, and applies gradient descent on
    dummy images and labels to minimize these differences. It also incorporates regularization through total variation
    (TV) norm to stabilize the optimization process.

    Args:
        number_of_workers (int): The number of workers participating in the computation.
        batch_dummy_image (list): A list of dummy images for each worker, where each element is a PyTorch `Parameter`.
        batch_dummy_label (Tensor): The dummy labels corresponding to the dummy images, as a PyTorch `Parameter`.
        local_net (list): A list of local network models for each worker.
        server (nn.Module): The server network model that processes the aggregated outputs from the workers.
        real_gradient (list): A list of real gradients obtained from the victim model for comparison.
        real_middle_input (list): A list of real middle-layer inputs obtained from the victim model for comparison.

    Returns:
        tuple:
            - D (float): The computed distance between real and fake gradients.
            - cafe_gradient_x (list): A list of gradients for the dummy images, used for updating the dummy images.
            - cafe_gradient_y (Tensor): The gradient for the dummy labels, used for updating the dummy labels.

    Example:
        D, gradient_x, gradient_y = cafe_torch(4, batch_dummy_image, batch_dummy_label, local_net, server, real_gradient, real_middle_input)
        This will return the gradient distance and gradients for further optimization.
    """

    fake_gradient = []
    fake_local_output = []
    fake_middle_input = []
    batch_dummy_label = nn.Parameter(batch_dummy_label)
    for worker_index in range(number_of_workers):
        batch_dummy_image[worker_index] = nn.Parameter(batch_dummy_image[worker_index])
        temp_middle_input, temp_local_output, temp_middle_output = local_net[
            worker_index
        ](batch_dummy_image[worker_index], cafe=True)
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
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between real and dummy images for each worker.

    PSNR is used to measure the similarity between two images. The higher the PSNR, the more similar
    the images are. This function computes the PSNR value for each worker and returns the aggregated
    PSNR value across all workers.

    Args:
        batch_real_image (list): A list of real images for each worker. Each image is a PyTorch tensor.
        batch_dummy_image (list): A list of dummy images for each worker. Each image is a PyTorch tensor.

    Returns:
        float: The aggregated PSNR value across all workers, representing the average similarity between
        the real and dummy images.

    Example:
        psnr_value = PSNR(batch_real_image, batch_dummy_image)
        This will compute and return the PSNR value.
    """
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
    """
    Evaluates the model on test data by computing the loss and accuracy.

    This function processes the test data through the local networks of each worker and the server network,
    computes the prediction, and evaluates the performance using loss and accuracy metrics.

    Args:
        number_of_workers (int): The number of workers participating in the evaluation.
        test_data (list): A list of test data for each worker, where each element is a PyTorch tensor.
        test_labels (Tensor): The true labels for the test data, as a PyTorch tensor.
        local_net (list): A list of local network models for each worker.
        server (nn.Module): The server network model that processes the aggregated outputs from the workers.

    Returns:
        tuple:
            - loss (Tensor): The computed loss between the predicted and true labels.
            - test_acc (float): The computed accuracy of the predictions.

    Example:
        loss, test_acc = test(4, test_data, test_labels, local_net, server)
        This will return the loss and accuracy for the test data.
    """
    local_output = []
    test_labels = test_labels
    for worker_index in range(number_of_workers):
        test_data[worker_index] = test_data[worker_index].float()
        temp_middle_input, temp_local_output, temp_middle_output = local_net[
            worker_index
        ](test_data[worker_index], cafe=True)
        local_output.append(temp_local_output)
    real_local_output = torch.cat(local_output, axis=1)  # batch size x 40
    predict = server(real_local_output)
    loss = compute_loss(test_labels, predict)
    test_acc = compute_accuracy(test_labels, predict)
    print(f"loss: {loss}, test_acc: {test_acc}")
    return loss, test_acc


def cafe_attack(
    local_net,
    server,
    dummy_data,
    dummy_labels,
    iter_num: int,
    dummy_index: list = None,
    batch_size: int = 40,
    data_number: int = 800,
    real_data: list = None,
    real_labels=None,
    true_gradient_list: list = None,
    number_of_workers: int = 4,
    dummy_middle_output_gradient=None,
    learning_rate_first_shot: float = 5e-3,
    dummy_middle_input=None,
    learning_rate_double_shot: float = 1e-2,
    cafe_learning_rate: float = 0.01,
    learning_rate_fl: float = 1e-6,
):
    model_list = [local_net[worker_index] for worker_index in range(number_of_workers)]
    model_list.append(server)

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
    print("\n", "-" * 30)
    random_lists = dummy_index

    batch_real_data = real_data
    true_gradient = true_gradient_list

    server.zero_grad()
    # server.zero_grad()
    for worker_index in range(number_of_workers):
        local_net[worker_index].zero_grad()

    batch_dummy_middle_output_gradient = take_batch(
        number_of_workers, dummy_middle_output_gradient, random_lists
    )
    optimizer = SGD(
        [{"params": [param]} for param in batch_dummy_middle_output_gradient],
        lr=learning_rate_first_shot,
    )

    for g_epoch in range(10):
        optimizer.zero_grad()
        for worker_index in range(number_of_workers):
            loss = (
                torch.norm(
                    torch.sum(batch_dummy_middle_output_gradient[worker_index], dim=0)
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
                        torch.transpose(batch_dummy_middle_input[worker_index], 0, 1),
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
            iter_num,
            batch_size,
            random_lists,
            g_middle_inputs,
            batch_dummy_middle_input,
        )

    for worker_index in range(number_of_workers):

        dummy_middle_input[worker_index].data[random_lists] = batch_dummy_middle_input[
            worker_index
        ].detach()
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
        iter_num, random_lists, cafe_gradient_x, batch_dummy_data
    )
    batch_dummy_label = opt_cafe.apply_gradients_label(
        iter_num, random_lists, cafe_gradient_y, batch_dummy_label
    )
    for worker_index in range(number_of_workers):
        dummy_data[worker_index].data[random_lists] = batch_dummy_data[
            worker_index
        ].detach()
        dummy_labels.data[random_lists] = batch_dummy_label[worker_index].detach()

    psnr = PSNR(batch_real_data, batch_dummy_data)

    if iter_num % 100 == 0:
        loss, test_acc = test(
            number_of_workers, real_data, real_labels, local_net, server
        )
        print(f"D: {D}, psnr: {psnr}, iter: {iter_num}, test_acc: {test_acc}")

    return dummy_data, dummy_labels, dummy_middle_output_gradient, dummy_middle_input
