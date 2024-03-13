#!/usr/bin/env python
# coding=utf-8
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

def forward_replacement(
    embeddings, batch_indexes, poisoning_indexes, target_indexes, blurred=False
):
    replacement_map = []
    poisoning_set = []
    target_set = []

    for i, bindex in enumerate(batch_indexes):
        if bindex in poisoning_indexes:
            poisoning_set.append(i)
        if (bindex in target_indexes) and (bindex not in poisoning_indexes):
            target_set.append(i)

    if blurred:
        for i in poisoning_set:
            embeddings[i] = torch.randn(embeddings[i].shape).to(embeddings[i].device)

    if len(target_set) > 0 and len(poisoning_set) > 0:
        for i in target_set:
            j = np.random.choice(poisoning_set)
            replacement_map.append((i, j))
            embeddings[i] = embeddings[j]

    return embeddings, replacement_map


def backward_replacement(gradients, replacement_map, gamma):
    for i, j in replacement_map:
        gradients[j] = gamma * gradients[i]

    return gradients
