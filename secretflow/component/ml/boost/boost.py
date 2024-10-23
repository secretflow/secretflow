# Copyright 2024 Ant Group Co., Ltd.
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

from collections import Counter


def get_party_features_info(
    meta,
) -> tuple[dict[str, list[str]], dict[str, tuple[int, int]]]:
    party_features_length: dict[str, int] = meta["party_features_length"]
    feature_names = meta["feature_names"]
    party_features_name: dict[str, list[str]] = dict()
    party_features_pos: dict[str, tuple[int, int]] = dict()
    party_pos = 0
    for party, f_len in party_features_length.items():
        party_features = feature_names[party_pos : party_pos + f_len]
        party_features_name[party] = party_features
        party_features_pos[party] = (party_pos, party_pos + f_len)
        party_pos += f_len

    return party_features_name, party_features_pos


def build_tree_attrs(
    node_ids, split_feature_indices, split_values, tree_leaf_indices=None
):
    assert len(node_ids) > 0, f"Too few nodes to form a tree structure."

    lchild_ids = [idx * 2 + 1 for idx in node_ids]
    rchild_ids = [idx * 2 + 2 for idx in node_ids]

    def _deal_leaf_node(
        child_node_id, node_ids, leaf_node_ids, split_feature_indices, split_values
    ):
        if child_node_id not in node_ids:
            # add leaf node
            leaf_node_ids.append(child_node_id)
            split_feature_indices.append(-1)
            split_values.append(0)

    leaf_node_ids = []
    for child_pos in range(len(lchild_ids)):
        _deal_leaf_node(
            lchild_ids[child_pos],
            node_ids,
            leaf_node_ids,
            split_feature_indices,
            split_values,
        )
        _deal_leaf_node(
            rchild_ids[child_pos],
            node_ids,
            leaf_node_ids,
            split_feature_indices,
            split_values,
        )

    for _ in range(len(leaf_node_ids)):
        lchild_ids.append(-1)
        rchild_ids.append(-1)

    node_ids.extend(leaf_node_ids)
    assert (
        len(node_ids) == len(lchild_ids)
        and len(node_ids) == len(rchild_ids)
        and len(node_ids) == len(split_feature_indices)
        and len(node_ids) == len(split_values)
    ), f"len of node_ids lchild_ids rchild_ids leaf_flags split_feature_indices split_values mismatch, {len(node_ids)} vs {len(lchild_ids)} vs {len(rchild_ids)} vs {len(split_feature_indices)} vs {len(split_values)}"

    if tree_leaf_indices is not None:
        assert Counter(leaf_node_ids) == Counter(
            tree_leaf_indices
        ), f"`leaf_node_ids`({leaf_node_ids}) and `tree_leaf_indices`({tree_leaf_indices}) do not have the same elements."
        return (
            node_ids,
            lchild_ids,
            rchild_ids,
            split_feature_indices,
            split_values,
            tree_leaf_indices,
        )
    else:
        return (
            node_ids,
            lchild_ids,
            rchild_ids,
            split_feature_indices,
            split_values,
            leaf_node_ids,
        )
