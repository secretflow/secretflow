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


import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def precision_recall(output, target):
    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    y_true = np.array(target.clone().detach().cpu())
    y_pred = np.array(pred.clone().detach().cpu()[0])
    if sum(y_pred) == 0:
        y_pred = np.ones_like(y_pred)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1.0:
                TP_samples_num += 1
            else:
                TN_samples_num += 1
            right_samples_num += 1
        else:
            if y_pred[i] == 1.0:
                FP_samples_num += 1
            else:
                FN_samples_num += 1
            wrong_samples_num += 1

    if (TP_samples_num + FP_samples_num) != 0:
        precision = TP_samples_num / (TP_samples_num + FP_samples_num)
    else:
        precision = 0
    if (TP_samples_num + FN_samples_num) != 0:
        recall = TP_samples_num / (TP_samples_num + FN_samples_num)
    else:
        recall = 0

    return precision, recall


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
