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

import warnings
from typing import Callable, List

from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric

from secretflow.ml.nn.core import torch as T


class BaseModule(T.BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            """Use of secretflow.ml.nn.utils.BaseModule is deprecated.
                Please use secretflow.ml.nn.core.torch.BaseModule instead.""",
            DeprecationWarning,
            stacklevel=2,
        )


class TorchModel(T.TorchModel):
    def __init__(
        self,
        model_fn: Callable[..., BaseModule] = None,
        loss_fn: Callable[..., BaseTorchLoss] = None,
        optim_fn: Callable[..., Optimizer] = None,
        metrics: List[Callable[..., Metric]] = [],
        **kwargs,
    ):
        super().__init__(model_fn, loss_fn, optim_fn, metrics, **kwargs)
        warnings.warn(
            """Use of secretflow.ml.nn.utils.TorchModel is deprecated.
                Please use secretflow.ml.nn.core.torch.TorchModel instead.""",
            DeprecationWarning,
            stacklevel=2,
        )


# TorchModel related utils here are deprecated
metric_wrapper = T.metric_wrapper
optim_wrapper = T.optim_wrapper
loss_wrapper = T.loss_wrapper


def plot_with_tsne(y_pred, eval_y, file_name):
    """
    Helper function to plot the t-SNE figure of output posteriors for nodes.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

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
