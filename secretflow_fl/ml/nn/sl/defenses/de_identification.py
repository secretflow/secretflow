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

import random

import numpy as np
import torch
from torch import nn

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

except ImportError:
    # PyTorch 1.6.0 and older versions

    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


class Maskinglayer(torch.nn.Module):
    """This is a plug and play layer for data de-identification. It contains no trainable parameters but some random state.
    It can be inserted into any part of the network and it does NOT change the shape of the input tensor. However, it implents
    some perturbations on the value of the tensor and you should retrain your network after using inserting this layer. The layer
    may influence the performance of the model, but it can be solved by data augmentation.
    Args:
        input_dim: the feature dimension which would be divided.
        subset_num: The total number of subsets into which the feature channels are divided.
        random_state: (Optional) random parameters utilized in the layer if provided.
    """

    def __init__(self, input_dim, subset_num, random_state=None) -> None:
        assert subset_num > 1
        super().__init__()
        self.input_dim = input_dim
        self.subset_num = subset_num
        self.generate_random_state()
        if random_state is not None:
            self.load_random_state(random_state)

    def dct(self, x, norm=None):
        """
        Cited by torch_dct
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = dct_fft_impl(v)

        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X, norm=None):
        """
        Cited by torch_dct
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct(dct(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = (
            torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
            * np.pi
            / (2 * N)
        )
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = idct_irfft_impl(V)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, : N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, : N // 2]

        return x.view(*x_shape)

    def generate_random_state(self):
        self.random_index = torch.tensor([i for i in torch.randperm(self.input_dim)])
        self.random_subsets = torch.chunk(self.random_index, self.subset_num, dim=0)
        self.selection_feature = []
        for subset_i in range(len(self.random_subsets)):
            vector_index = nn.Parameter(
                torch.zeros(self.input_dim), requires_grad=False
            )
            non_zero_f = sorted(self.random_subsets[subset_i])
            for j in range(len(non_zero_f)):
                channel = non_zero_f[j]
                vector_index[channel] = 1
            self.selection_feature.append(vector_index)
        self.selection_feature = nn.Parameter(
            torch.stack(self.selection_feature, dim=0), requires_grad=False
        )
        self.shuffle_index1 = nn.Parameter(
            torch.randperm(self.input_dim), requires_grad=False
        )
        self.shuffle_index2 = nn.Parameter(
            torch.randperm(self.input_dim), requires_grad=False
        )
        self.matrix = torch.randn(self.input_dim, self.input_dim)
        self.q, self.r = torch.linalg.qr(self.matrix)
        self.q[:, -1] = 0
        self.q = nn.Parameter(self.q, requires_grad=False)

    def load_random_state(self, random_state_path):
        random_state = torch.load(random_state_path)
        self.load_state_dict(random_state)

    def emb_pp(self, batch_data, subset):
        batch_data = batch_data[:, self.shuffle_index1]
        batch_dct = self.dct(batch_data)
        batch_dct_p = batch_dct * self.selection_feature[subset]
        batch_idct = self.idct(batch_dct_p)
        batch_x_pp = torch.mm(batch_idct, self.q)
        batch_x_pp = batch_x_pp[:, self.shuffle_index2]
        return batch_x_pp

    def forward(self, x, subset=None):
        x_size = x.shape
        if subset is None:
            subset = [random.randint(0, self.subset_num - 1) for _ in range(x_size[0])]
        # PP_embed
        x_pp = self.emb_pp(x, subset)

        return x_pp
