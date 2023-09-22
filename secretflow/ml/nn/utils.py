#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2022 Ant Group Co., Ltd.
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


from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torchmetrics import Metric


class BaseModule(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def get_weights(self, return_numpy=False):
        if not return_numpy:
            return {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            weights_list = []
            for v in self.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def update_weights(self, weights):
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(np.copy(v))
        self.load_state_dict(weights_dict)

    def get_gradients(self, parameters=None):
        if parameters is None:
            parameters = self.parameters()
        grads = []
        for p in parameters:
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return [g.copy() for g in grads]

    def set_gradients(
        self,
        gradients: List[Union[torch.Tensor, np.ndarray]],
        parameters: Optional[List[torch.Tensor]] = None,
    ):
        if parameters is None:
            parameters = self.parameters()
        for g, p in zip(gradients, parameters):
            if g is not None:
                p.grad = torch.from_numpy(np.array(g.copy()))


class TorchModel:
    def __init__(
        self,
        model_fn: BaseModule = None,
        loss_fn: BaseTorchLoss = None,
        optim_fn: optim.Optimizer = None,
        metrics: List[Metric] = [],
        **kwargs,
    ):
        self.model_fn = model_fn
        self.loss_fn: BaseTorchLoss = loss_fn
        self.optim_fn: optim.Optimizer = optim_fn
        self.metrics: List[Metric] = metrics
        self.kwargs = kwargs


def metric_wrapper(func, *args, **kwargs):
    def wrapped_func():
        return func(*args, **kwargs)

    return wrapped_func


def optim_wrapper(func, *args, **kwargs):
    def wrapped_func(params):
        return func(params, *args, **kwargs)

    return wrapped_func


def plot_with_tsne(y_pred, eval_y, file_name):
    """
    Helper function to plot the t-SNE figure of output posteriors for nodes.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def plot_with_labels(lowDWeights, labels):
        plt.cla()
        X, Y = (
            lowDWeights[:, 0],
            lowDWeights[:, 1],
        )
        plt.scatter(X, Y, c=labels, label='t-SNE')
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.title(f'Visualize last layer - {file_name}')
        import os

        plt.savefig(f"{os.curdir}/{file_name}.pdf")

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_num = 666
    low_dim_embs = tsne.fit_transform(y_pred[:plot_num, :])
    labels = eval_y.argmax(1)[:plot_num]
    plot_with_labels(low_dim_embs, labels)
