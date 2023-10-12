# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 13:34
@Author: KI
@File: ScaffoldOptimizer.py
@Motto: Hungry And Humble
"""
from torch.optim import Optimizer
import torch
import numpy as np


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay=0.5):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)
        self.lr=lr

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                dp = p.grad.data + np.array(c) - np.array(ci)
                dp = p.grad.data 
                p.data = p.data - dp.data * group['lr']

        return loss
    
    def get_grad(self):
        grad_=[]
        for group in self.param_groups:
            for p  in group['params']:
                if p.grad is None:
                    continue
                grad_.append(p.grad.data.numpy())
        return grad_