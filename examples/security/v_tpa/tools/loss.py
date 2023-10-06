#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F

def entropy(pred):
    return torch.mean(torch.sum(-pred * torch.log2(pred + 1e-30), 1))

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def cross_entropy(pred, target):
    return torch.mean(torch.sum(-target * torch.log(pred + 1e-30), 1))
