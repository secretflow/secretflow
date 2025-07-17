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
from typing import List, Union

import numpy as np
from secretflow_serving_lib.link_function_pb2 import LinkFunctionType
from secretflow_spec import VTableSchema

from secretflow.component.core.dist_data.vtable_utils import VTableUtils
from secretflow.component.core.serving_builder import (
    DispatchType,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
)
from secretflow.compute.tracer import Table
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.device.spu import SPUObject
from secretflow.device.kernels.spu import party_shards_to_heu_plain_text


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


class TreeNodeContext:
    def __init__(self, he_mode: bool = False, heu_dict=None):
        self.he_mode = he_mode
        self.heu_dict = heu_dict
        self.party_select_kwargs = dict()
        self.party_select_inputs = dict()
        self.party_select_outputs = dict()
        self.party_merge_kwargs = dict()
        self.party_merge_inputs = dict()
        self.party_merge_outputs = dict()
        self.party_specific_flag = dict()

    def add_party_attr(
        self,
        pyu: PYU,
        input_schema: VTableSchema,
        feature_names,
        label_col,
        node_ids,
        lchild_ids,
        rchild_ids,
        split_feature_indices,
        split_values,
        leaf_node_ids,
        weights: Union[SPUObject, PYUObject],
    ):
        # tree_select node
        self.party_select_inputs[pyu] = Table.from_schema(
            VTableUtils.to_arrow_schema(input_schema.select(feature_names))
        ).dump_serving_pb("tmp")[1]

        self.party_select_kwargs[pyu] = {
            "input_feature_names": feature_names,
            "input_feature_types": [
                VTableUtils.to_serving_dtype(input_schema[f].type)
                for f in feature_names
            ],
            "root_node_id": 0,
            "node_ids": node_ids,
            "lchild_ids": lchild_ids,
            "rchild_ids": rchild_ids,
            "split_feature_idxs": split_feature_indices,
            "split_values": split_values,
            "leaf_node_ids": leaf_node_ids,
        }

        if self.he_mode:
            assert isinstance(weights, SPUObject)

            heu = self.heu_dict[pyu.party]
            heu_weight_shard = party_shards_to_heu_plain_text(weights, heu, pyu.party)
            pyu_weights = heu_weight_shard.serialize_to_pyu(pyu)

            self.party_select_kwargs[pyu]['select_col_name'] = 'selects'
            self.party_select_kwargs[pyu]['weight_shard_col_name'] = 'weight_shards'
            self.party_select_kwargs[pyu]['weight_shard'] = pyu_weights
            self.party_select_outputs[pyu] = Table.from_schema(
                {"selects": np.bytes_, "weight_shards": np.bytes_}
            ).dump_serving_pb("tmp")[1]

            self.party_merge_inputs[pyu] = self.party_select_outputs[pyu]
            self.party_merge_outputs[pyu] = Table.from_schema(
                {"tree_score": np.bytes_}
            ).dump_serving_pb("tmp")[1]
            self.party_merge_kwargs[pyu] = {
                "select_col_name": "selects",
                "weight_shard_col_name": "weight_shards",
                "output_col_name": "tree_score",
                "weight_shard": pyu_weights,
            }
        else:
            self.party_select_kwargs[pyu]['output_col_name'] = 'selects'
            self.party_select_outputs[pyu] = Table.from_schema(
                {"selects": np.uint64}
            ).dump_serving_pb("tmp")[1]

            # tree_merge node
            self.party_merge_inputs[pyu] = self.party_select_outputs[pyu]
            self.party_merge_outputs[pyu] = Table.from_schema(
                {"weights": np.float64}
            ).dump_serving_pb("tmp")[1]

            self.party_merge_kwargs[pyu] = {
                "input_col_name": "selects",
                "output_col_name": "weights",
            }
            self.party_specific_flag[pyu] = False
            if label_col in input_schema.fields:
                self.party_merge_kwargs[pyu]["leaf_weights"] = pyu(lambda w: list(w))(
                    weights.to(pyu) if isinstance(weights, SPUObject) else weights
                )
                self.party_specific_flag[pyu] = True


def build_tree_model(
    tree_node_ctxs: List[TreeNodeContext],
    serving_builder: ServingBuilder,
    model_type: str,
    node_id,
    pyus,
    parties,
    pred_name: str,
    tree_num: int,
    algo_func: LinkFunctionType,
    base_score,
    parent_node_name: str,
    he_mode: bool,
):
    tree_select_names = []
    for pos, ctx in enumerate(tree_node_ctxs):
        select_node_name = (
            f"phe_{model_type}_{node_id}_select_{pos}"
            if he_mode
            else f"{model_type}_{node_id}_select_{pos}"
        )
        op = ServingOp.PHE_2P_TREE_SELECT if he_mode else ServingOp.TREE_SELECT
        node = ServingNode(
            select_node_name,
            op,
            ServingPhase.TRAIN_PREDICT,
            ctx.party_select_inputs,
            ctx.party_select_outputs,
            ctx.party_select_kwargs,
            [parent_node_name] if parent_node_name else [],
        )
        serving_builder.add_node(node)
        tree_select_names.append(select_node_name)

    def _finish_merge_node(
        tree_node_ctxs: List[TreeNodeContext],
        serving_builder: ServingBuilder,
        model_type,
        he_mode,
        node_id,
    ):
        node_names = []
        for pos, ctx in enumerate(tree_node_ctxs):
            n_name = (
                f"phe_{model_type}_{node_id}_merge_{pos}"
                if he_mode
                else f"{model_type}_{node_id}_merge_{pos}"
            )
            parents = [tree_select_names[pos]]
            assert len(parents) == 1
            node = ServingNode(
                n_name,
                ServingOp.PHE_2P_TREE_MERGE if he_mode else ServingOp.TREE_MERGE,
                ServingPhase.TRAIN_PREDICT,
                ctx.party_merge_inputs,
                ctx.party_merge_outputs,
                ctx.party_merge_kwargs,
                parents,
            )
            serving_builder.add_node(node)
            node_names.append(n_name)
        return node_names

    if he_mode:
        serving_builder.new_execution(DispatchType.DP_PEER)

        # add tree_merge nodes
        ensemble_merge_parent_node_names = _finish_merge_node(
            tree_node_ctxs, serving_builder, model_type, he_mode, node_id
        )

        party_ensemble_merge_inputs = dict()
        party_ensemble_merge_outputs = dict()
        party_ensemble_merge_kwargs = dict()
        party_ensemble_predict_inputs = dict()
        party_ensemble_predict_outputs = dict()
        party_ensemble_predict_kwargs = dict()

        for party in parties:
            # tree_ensemble_merge
            party_ensemble_merge_inputs[pyus[party]] = tree_node_ctxs[
                0
            ].party_merge_outputs[pyus[party]]
            party_ensemble_merge_outputs[pyus[party]] = Table.from_schema(
                {"encrypted_score": np.bytes_}
            ).dump_serving_pb("tmp")[1]
            party_ensemble_merge_kwargs[pyus[party]] = {
                "input_col_name": "tree_score",
                "output_col_name": "encrypted_score",
                "num_trees": tree_num,
            }
            # tree_ensemble_predict
            party_ensemble_predict_inputs[pyus[party]] = party_ensemble_merge_outputs[
                pyus[party]
            ]
            party_ensemble_predict_outputs[pyus[party]] = Table.from_schema(
                {pred_name: np.float64}
            ).dump_serving_pb("tmp")[1]
            party_ensemble_predict_kwargs[pyus[party]] = {
                "input_col_name": "encrypted_score",
                "output_col_name": pred_name,
                "algo_func": algo_func,
                "base_score": base_score,
            }

        # add tree_ensemble_merge node
        node = ServingNode(
            f"phe_{model_type}_{node_id}_ensemble_merge",
            ServingOp.PHE_2P_TREE_ENSEMBLE_MERGE,
            ServingPhase.TRAIN_PREDICT,
            party_ensemble_merge_inputs,
            party_ensemble_merge_outputs,
            party_ensemble_merge_kwargs,
            ensemble_merge_parent_node_names,
        )
        serving_builder.add_node(node)

        serving_builder.new_execution(DispatchType.DP_SELF)

        # add tree_ensemble_predict node
        node = ServingNode(
            f"phe_{model_type}_{node_id}_ensemble_predict",
            ServingOp.PHE_2P_TREE_ENSEMBLE_PREDICT,
            ServingPhase.TRAIN_PREDICT,
            party_ensemble_predict_inputs,
            party_ensemble_predict_outputs,
            party_ensemble_predict_kwargs,
        )
        serving_builder.add_node(node)
    else:
        serving_builder.new_execution(
            DispatchType.DP_SPECIFIED,
            party_specific_flag=tree_node_ctxs[0].party_specific_flag,
        )

        # add tree_merge nodes
        predict_parent_node_names = _finish_merge_node(
            tree_node_ctxs, serving_builder, model_type, he_mode, node_id
        )

        # add tree_ensemble_predict node
        party_predict_inputs = dict()
        party_predict_outputs = dict()
        party_predict_kwargs = dict()
        for party in parties:
            party_predict_inputs[pyus[party]] = tree_node_ctxs[0].party_merge_outputs[
                pyus[party]
            ]
            party_predict_outputs[pyus[party]] = Table.from_schema(
                {pred_name: np.float64}
            ).dump_serving_pb("tmp")[1]
            party_predict_kwargs[pyus[party]] = {
                "input_col_name": "weights",
                "output_col_name": pred_name,
                "algo_func": algo_func,
                "num_trees": tree_num,
                "base_score": base_score,
            }

        node = ServingNode(
            f"{model_type}_{node_id}_predict",
            ServingOp.TREE_ENSEMBLE_PREDICT,
            ServingPhase.TRAIN_PREDICT,
            party_predict_inputs,
            party_predict_outputs,
            party_predict_kwargs,
            predict_parent_node_names,
        )
        serving_builder.add_node(node)
