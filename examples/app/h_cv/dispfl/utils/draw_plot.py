# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys

import matplotlib.pyplot as plt

dispfl_root_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, dispfl_root_path)


class ACC:
    def __init__(self):
        self.round = []
        self.acc = []
        self.loss = []

    def append(self, results: list):
        loss = float(results[2])
        if len(results) == 4:  # there are 1.23111e-04 expression
            loss = loss / (10 ** int(results[3]))

        self.round.append(float(results[0]))
        self.acc.append(float(results[1]))
        self.loss.append(loss)


def extract_acc(path, name=None) -> ACC:
    """extract a list with communication rounds (x) and acc (y) for plot."""
    if name is None:
        name = path[:-3]
    pattern = r"INFO:myLogger:In round \d+ after local trainning, test_acc = \d+\.?\d*, test_loss = \d+\.?\d*([\s\S]*)"
    num_pattern = "\d+\.?\d*"
    with open(path, "r") as f:
        acc = ACC()
        for line in f:
            line = line.rstrip()
            if re.match(pattern, line) is not None:
                acc.append(re.findall(num_pattern, line))
        return acc


def draw_acc(total_rows, total_cols, cur_idx: int, plain_acc: ACC, sf_acc: ACC, title):
    """draw a subplot
    Args:
        total_rows: r * l plots
        total_cols: r * l plots
        cur_idx: current draw idx
        plain_acc: plain computation acc
        sf_acc: secretflow computation acc
        title: a title of this picture
    """
    ax = plt.subplot(total_rows, total_cols, cur_idx)
    (l1,) = plt.plot(
        plain_acc.round,
        plain_acc.acc,
        color='r',
        marker=None,
        linestyle="-",
        linewidth=0.5,
    )
    (l2,) = plt.plot(
        sf_acc.round, sf_acc.acc, color='b', marker=None, linestyle='-', linewidth=0.5
    )
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)
    plt.xlabel("Communication rounds", fontsize=18)
    plt.ylabel("Test Accuracy", fontsize=18)
    plt.legend(
        handles=[l1, l2], labels=['plain', 'secretflow'], loc=4
    )  # loc = best for best
    plt.title(title, fontsize=18)


def run_drawing():
    # change this log with your logs.
    candidates = [
        ("large_plain_cifar10_500.log", "large_sf_cifar10_500.log"),
        ("large_plain_cifar100_500.log", "large_sf_cifar100_500.log"),
        ("large_plain_cifar10_500_ncls.log", "large_sf_cifar10_500_ncls.log"),
        ("large_plain_cifar100_500_ncls.log", "large_sf_cifar100_500_ncls.log"),
    ]
    # change this directory to your log directory.
    directory = "../../log/all/"
    fig = plt.figure(1, figsize=(20, 15))  # generate figure
    total_rows = 2
    total_cols = 2
    for i, candidate in enumerate(candidates):
        plain = candidate[0]
        sf = candidate[1]
        cifar = "cifar100" if "cifar100" in plain else "cifar10"
        load_type = "path-part" if "ncls" in plain else "dir-part"
        title = cifar + "-" + load_type
        plain_acc_results = extract_acc(directory + plain)
        sf_acc_results = extract_acc(directory + sf)
        draw_acc(
            total_rows, total_cols, (i + 1), plain_acc_results, sf_acc_results, title
        )
    plt.show()


if __name__ == '__main__':
    run_drawing()
