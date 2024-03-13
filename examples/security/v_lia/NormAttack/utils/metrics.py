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

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F


def total_variance(x):
    """Returns the total variance of the given data

    Args:
        x (torch.Tensor): input data

    Returns:
        float: total variance of the given data
    """
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def crossentropyloss_between_logits(y_pred_logit, y_true_labels, reduction="mean"):
    """Cross entropy loss for soft labels
    Based on https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2
    Args:
        y_pred_logit (torch.Tensor): predicted logits
        y_true_labels (torch.Tensor): ground-truth soft labels

    Returns:
        torch.Tensor: average cross entropy between y_pred_logit and y_true_labels2
    """
    results = -torch.sum(F.log_softmax(y_pred_logit, dim=1) * y_true_labels, dim=1)

    if reduction == "sum":
        return torch.sum(results)
    elif reduction == "mean":
        return torch.mean(results)
    else:
        raise NotImplementedError(f"`reduction`={reduction} is not supported.")


def accuracy_torch_dataloader(model, dataloader, device="cpu", xpos=1, ypos=2):
    """Calculates the accuracy of the model on the given dataloader

    Args:
        model (torch.nn.Module): model to be evaluated
        dataloader (torch.DataLoader): dataloader to be evaluated
        device (str, optional): device type. Defaults to "cpu".
        xpos (int, optional): the positional index of the input in data. Defaults to 1.
        ypos (int, optional): the positional index of the label in data. Defaults to 2.

    Returns:
        float: accuracy
    """
    in_preds = []
    in_label = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data[xpos]
            labels = data[ypos]
            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.int64)
            outputs = model(inputs)
            in_preds.append(outputs)
            in_label.append(labels)
        in_preds = torch.cat(in_preds)
        in_label = torch.cat(in_label)

    return accuracy_score(
        np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label.cpu())
    )
