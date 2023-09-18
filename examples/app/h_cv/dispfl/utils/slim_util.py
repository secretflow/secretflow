# MIT License
#
# Copyright (c) 2022 Rong Dai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch


def cosine_annealing(args, round):
    return args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / args.comm_round))


def model_difference(model_a, model_b):
    a = sum(
        [torch.sum(torch.square(model_a[name] - model_b[name])) for name in model_a]
    )
    return a


def hamming_distance(mask_a, mask_b):
    dis = 0
    total = 0
    for key in mask_a:
        dis += torch.sum(mask_a[key].int() ^ mask_b[key].int())
        total += mask_a[key].numel()
    return dis, total
