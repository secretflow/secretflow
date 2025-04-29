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

import copy
import random
import logging
import numpy as np
import torch

from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder


def poison_dataset(dataloader, poison_rate, target_label):
    # Retrieve the dataset from the DataLoader and compute its length
    dataset = dataloader.dataset
    len_dataset = len(dataset)
    # Select a random sample of indices to poison based on the poison rate
    poison_index = random.sample(range(len_dataset), k=int(len_dataset * poison_rate))

    batch_size = dataloader.batch_size
    # Prepare the target label by one-hot encoding it
    target_label = np.array([target_label])
    classes = np.arange(10).reshape(-1, 1)
    encoder = OneHotEncoder(categories="auto", sparse_output=False)
    encoder.fit(classes)
    target_label = encoder.transform(target_label.reshape(-1, 1))
    target_label = np.squeeze(target_label)

    x = []
    y = []
    patch_size = 4  # The size of the patch region to modify in the frequency domain

    for index in range(len_dataset):
        tmp = list(dataset[index])
        if index in poison_index:
            # Convert the image tensor to a NumPy array; assumed shape is (C, H, W)
            img = tmp[0].numpy()
            img_poisoned = np.empty_like(img)
            for c in range(img.shape[0]):
                channel = img[c]
                # Compute the Fourier transform of the current channel
                f = np.fft.fft2(channel)
                fshift = np.fft.fftshift(f)
                # Modify a patch in the center region of the frequency domain
                centerH, centerW = fshift.shape[0] // 2, fshift.shape[1] // 2
                fshift[
                    centerH - patch_size // 2 : centerH + patch_size // 2,
                    centerW - patch_size // 2 : centerW + patch_size // 2,
                ] = 10.0
                # Inverse transform back to the spatial domain
                f_ishift = np.fft.ifftshift(fshift)
                channel_back = np.fft.ifft2(f_ishift)
                # Take the real part as the modified channel data
                img_poisoned[c] = np.real(channel_back)
            # Convert the modified image back to a tensor and replace the original image
            tmp[0] = torch.from_numpy(img_poisoned.astype(np.float32))
            # Replace the label with the target label
            tmp[1] = target_label
        else:
            # For benign samples, convert the label to numpy array
            tmp[1] = tmp[1].numpy()

        x.append(tmp[0].numpy())
        y.append(tmp[1])

    x = np.array(x)
    assert x.dtype == np.float32
    y = np.array(y)
    y = np.squeeze(y)
    data_list = [torch.Tensor((x.astype(np.float64)).copy())]
    data_list.append(torch.Tensor(y.astype(np.float64).copy()))
    dataset = TensorDataset(*data_list)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )
    assert len(train_loader) > 0
    return train_loader


class BackdoorAttack(AttackCallback):
    def __init__(
        self,
        attack_party: PYU,
        poison_rate: float = 0.1,
        target_label: int = 1,
        eta: float = 1.0,
        attack_epoch: int = 50,
    ):
        super().__init__()
        self.attack_party = attack_party
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.eta = eta
        self.attack_epoch = attack_epoch

    def on_epoch_begin(self, epoch):
        if epoch == self.attack_epoch:

            def init_attacker_worker(attack_worker, poison_rate, target_label):
                attack_worker.benign_train_set = copy.deepcopy(attack_worker.train_set)
                attack_worker.train_set = poison_dataset(
                    attack_worker.train_set, poison_rate, target_label
                )

            self._workers[self.attack_party].apply(
                init_attacker_worker, self.poison_rate, self.target_label
            )

    def on_train_batch_inner_before(self, epoch, device):
        def attacker_model_initial(attack_worker):
            attack_worker.init_weights = attack_worker.get_weights(return_numpy=True)

        if device == self.attack_party:
            self._workers[self.attack_party].apply(attacker_model_initial)
