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

"""Utility functions for training.
"""
import torch
from torch.optim.optimizer import Optimizer

###################################
#            Optimizers           #
###################################


class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an
    initial accumulater value. This mimics the behavior of the default Adagrad
    implementation in Tensorflow. The default PyTorch Adagrad uses 0 for
    initial acculmulator value.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 1e-2)
        lr_decay (float, optional): Learning rate decay (default: 0)
        init_accu_value (float, optional): Initial accumulater value.
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0
    ):
        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            init_accu_value=init_accu_value,
            weight_decay=weight_decay,
        )
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["sum"] = (
                    torch.ones(p.data.size()).type_as(p.data) * init_accu_value
                )

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with "
                            "sparse gradients"
                        )
                    grad = grad.add(group["weight_decay"], p.data)

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                if p.grad.data.is_sparse:
                    # The update is non-linear so indices must be unique
                    grad = grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)

                    state["sum"].add_(make_sparse(grad_values.pow(2)))
                    std = state["sum"]._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state["sum"].addcmul_(1, grad, grad)
                    std = state["sum"].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss


def get_optimizer(name, parameters, lr, l2=0):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name in ["adagrad", "myadagrad"]:
        # Use my own adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == "adam":
        # Use default lr
        return torch.optim.Adam(parameters, weight_decay=l2, lr=lr, betas=(0.9, 0.98))
    elif name == "adamax":
        # Use default lr
        return torch.optim.Adamax(parameters, weight_decay=l2)
    elif name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


###################################
#          Training Utils         #
###################################


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    """Keeps only the topk rows of grads."""
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad
