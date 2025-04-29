#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


from dataclasses import dataclass


@dataclass()
class Node:
    """Tree Node

    Attributes:
        id: node id
        fid: feature id
        bid: bucket id
        weight: node weight
        is_leaf: whether this node is leaf
        sum_grad: sum of grad
        sum_hess: sum of hess
        left_nodeid: left node id
        right_nodeid: right node id
        missing_dir: which branch to go when encounting missing value default 1->right
        sample_num: num of data sample
        parent_nodeid: parent nodeid
        is_left_node: is this node if left child of the parent
        sibling_nodeid: sibling node id
        loss_change: the loss change.
    """

    id: int = None
    fid: int = None
    bid: int = None
    weight: float = 0.0
    is_leaf: bool = False
    sum_grad: float = None
    sum_hess: float = None
    left_nodeid: int = -1
    right_nodeid: int = -1
    missing_dir: int = 1
    sample_num: int = 0
    parent_nodeid: int = None
    is_left_node: bool = False
    sibling_nodeid: int = None
    loss_change: float = 0.0

    def __str__(self):
        return (
            f"id{self.id}, fid:{self.fid}, bid:{self.bid}, weight:{self.weight}, sum_grad:{self.sum_grad}, "
            f"sum_hess:{self.sum_hess}, left_node:{self.left_nodeid}, right_node:{self.right_nodeid}, "
            f"sample_num:{self.sample_num}, is_leaf:{self.is_leaf}, loss_change:{self.loss_change}"
        )
