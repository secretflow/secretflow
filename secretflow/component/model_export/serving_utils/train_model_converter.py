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


import json
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
from secretflow_serving_lib.link_function_pb2 import LinkFunctionType

import secretflow.component.ml.boost.ss_xgb.ss_xgb as xgb
import secretflow.component.ml.linear.ss_glm as glm
import secretflow.component.ml.linear.ss_sgd as sgd
from secretflow.component.data_utils import (
    DistDataType,
    extract_table_header,
    model_loads,
    model_meta_info,
)
from secretflow.component.ml.boost.sgb import sgb
from secretflow.component.ml.boost.ss_xgb.ss_xgb import build_ss_xgb_model
from secretflow.compute import Table
from secretflow.device import PYU, SPU, PYUObject, SPUObject
from secretflow.ml.boost.sgb_v.core.params import RegType as SgbRegType
from secretflow.ml.boost.ss_xgb_v.core.node_split import RegType as SSXgbRegType
from secretflow.ml.boost.ss_xgb_v.core.xgb_tree import XgbTree
from secretflow.ml.linear import RegType
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.utils.sigmoid import SigType

from .graph_builder_manager import GraphBuilderManager

SF_TYPE_TO_SERVING_TYPE = {
    np.int8: 'DT_INT8',
    np.int16: 'DT_INT16',
    np.int32: 'DT_INT32',
    np.int64: 'DT_INT64',
    np.uint8: 'DT_UINT8',
    np.uint16: 'DT_UINT16',
    np.uint32: 'DT_UINT32',
    np.uint64: 'DT_UINT64',
    np.float32: 'DT_FLOAT',
    np.float64: 'DT_DOUBLE',
    bool: 'DT_BOOL',
    int: 'DT_INT64',
    float: 'DT_DOUBLE',
    object: 'DT_STRING',
}

SS_GLM_LINK_MAP = {
    "Logit": LinkFunctionType.LF_SIGMOID_SR,
    "Log": LinkFunctionType.LF_EXP,
    "Reciprocal": LinkFunctionType.LF_RECIPROCAL,
    "Identity": LinkFunctionType.LF_IDENTITY,
}


SS_SGD_LINK_MAP = {
    SigType.REAL: LinkFunctionType.LF_SIGMOID_RAW,
    SigType.T1: LinkFunctionType.LF_SIGMOID_T1,
    SigType.T3: LinkFunctionType.LF_SIGMOID_T3,
    SigType.T5: LinkFunctionType.LF_SIGMOID_T5,
    SigType.DF: LinkFunctionType.LF_SIGMOID_DF,
    SigType.SR: LinkFunctionType.LF_SIGMOID_SR,
    SigType.MIX: LinkFunctionType.LF_SIGMOID_SEGLS,
}


def sf_type_to_serving_str(t: np.dtype) -> str:
    assert t in SF_TYPE_TO_SERVING_TYPE, f"not support type {t}"
    return SF_TYPE_TO_SERVING_TYPE[t]


def get_party_features_info(
    meta,
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[int, int]]]:
    party_features_length: Dict[str, int] = meta["party_features_length"]
    feature_names = meta["feature_names"]
    party_features_name: Dict[str, List[str]] = dict()
    party_features_pos: Dict[str, Tuple[int, int]] = dict()
    party_pos = 0
    for party, f_len in party_features_length.items():
        party_features = feature_names[party_pos : party_pos + f_len]
        party_features_name[party] = party_features
        party_features_pos[party] = (party_pos, party_pos + f_len)
        party_pos += f_len

    return party_features_name, party_features_pos


def reveal_to_pyu(
    spu: SPU, spu_w: SPUObject, start: int, end: int, to: PYU
) -> PYUObject:
    def _slice(w):
        w = w.reshape((-1, 1))
        return w.flatten()[start:end]

    sliced_w = spu(_slice)(spu_w)
    return to(lambda w: list(w))(sliced_w.to(to))


def linear_model_converter(
    party_features_name: Dict[str, List[str]],
    party_features_pos: Dict[str, Tuple[int, int]],
    input_schema: Dict[str, Dict[str, np.dtype]],
    feature_names: List[str],
    spu_w: SPUObject,
    label_col: str,
    offset_col: str,
    yhat_scale: float,
    link_type: LinkFunctionType,
    pred_name: str,
    traced_input: Dict[str, Set[str]],
):
    assert set(party_features_name).issubset(set(input_schema))
    assert set(party_features_name) == set(party_features_pos)
    assert len(party_features_name) > 0

    spu = spu_w.device
    party_dot_input_schemas = dict()
    party_dot_output_schemas = dict()
    party_merge_input_schemas = dict()
    party_merge_output_schemas = dict()
    party_dot_kwargs = dict()
    party_merge_kwargs = dict()
    for party, input_features in input_schema.items():
        pyu = PYU(party)

        assert party in traced_input

        if party in party_features_name:
            party_features = party_features_name[party]
            assert set(party_features).issubset(set(input_features))
            start, end = party_features_pos[party]
            pyu_w = reveal_to_pyu(spu, spu_w, start, end, pyu)
        else:
            party_features = []
            pyu_w = pyu(lambda: [])()

        if offset_col in input_features:
            party_features.append(offset_col)

            def append_one(w):
                w.append(1.0)
                return w

            pyu_w = pyu(append_one)(pyu_w)

        if label_col in input_features:
            intercept = reveal_to_pyu(
                spu, spu_w, len(feature_names), len(feature_names) + 1, pyu
            )
            intercept = pyu(lambda i: i[0])(intercept)
        else:
            intercept = 0

        assert set(party_features).issubset(traced_input[party])

        party_dot_input_schemas[pyu] = Table.from_schema(
            {f: input_features[f] for f in party_features}
        ).dump_serving_pb("tmp")[1]
        party_dot_output_schemas[pyu] = Table.from_schema(
            {"partial_y": np.float32}
        ).dump_serving_pb("tmp")[1]

        party_dot_kwargs[pyu] = {
            "feature_names": party_features,
            "feature_weights": pyu_w,
            "input_types": [
                sf_type_to_serving_str(input_features[f]) for f in party_features
            ],
            "output_col_name": "partial_y",
            "intercept": intercept,
        }

        party_merge_input_schemas[pyu] = Table.from_schema(
            {"partial_y": np.float32}
        ).dump_serving_pb("tmp")[1]
        party_merge_output_schemas[pyu] = Table.from_schema(
            {"pred_name": np.float32}
        ).dump_serving_pb("tmp")[1]

        party_merge_kwargs[pyu] = {
            "yhat_scale": yhat_scale,
            "link_function": LinkFunctionType.Name(link_type),
            "input_col_name": "partial_y",
            "output_col_name": pred_name,
        }

    return (
        party_dot_kwargs,
        party_merge_kwargs,
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_merge_input_schemas,
        party_merge_output_schemas,
    )


def ss_glm_schema_info(
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
):
    model_meta_str = model_meta_info(
        model_ds,
        glm.MODEL_MAX_MAJOR_VERSION,
        glm.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_MODEL,
    )
    meta = json.loads(model_meta_str)

    offset_col = meta["offset_col"]
    party_features_name, _ = get_party_features_info(meta)

    party_used_schemas = {}
    for party, input_features in input_schema.items():
        if party in party_features_name:
            used_schemas = party_features_name[party]
        else:
            used_schemas = []

        if offset_col in input_features:
            used_schemas.append(offset_col)

        party_used_schemas[party] = {"used": set(used_schemas)}

    return party_used_schemas


def ss_glm_converter(
    ctx,
    node_id: int,
    builder: GraphBuilderManager,
    spu_config,
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
    pred_name: str,
    traced_input: Dict[str, Set[str]],
):
    cluster_def = spu_config["cluster_def"].copy()
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40
    spu = SPU(cluster_def, spu_config["link_desc"])

    model_objs, model_meta_str = model_loads(
        ctx,
        model_ds,
        glm.MODEL_MAX_MAJOR_VERSION,
        glm.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_MODEL,
        spu=spu,
    )
    meta = json.loads(model_meta_str)
    spu_w = model_objs[0]

    feature_names = meta["feature_names"]
    party_features_name, party_features_pos = get_party_features_info(meta)
    offset_col = meta["offset_col"]
    label_col = meta["label_col"]
    yhat_scale = meta["y_scale"]
    assert len(label_col) == 1
    assert meta["link"] in SS_GLM_LINK_MAP
    label_col = label_col[0]

    (
        party_dot_kwargs,
        party_merge_kwargs,
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_merge_input_schemas,
        party_merge_output_schemas,
    ) = linear_model_converter(
        party_features_name,
        party_features_pos,
        input_schema,
        feature_names,
        spu_w,
        label_col,
        offset_col,
        yhat_scale,
        SS_GLM_LINK_MAP[meta["link"]],
        pred_name,
        traced_input,
    )

    builder.add_node(
        f"ss_glm_{node_id}_dot",
        "dot_product",
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_dot_kwargs,
    )
    builder.new_execution("DP_ANYONE")
    builder.add_node(
        f"ss_glm_{node_id}_merge_y",
        "merge_y",
        party_merge_input_schemas,
        party_merge_output_schemas,
        party_merge_kwargs,
    )


def ss_sgd_schema_info(
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
):
    model_meta_str = model_meta_info(
        model_ds,
        sgd.MODEL_MAX_MAJOR_VERSION,
        sgd.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_SGD_MODEL,
    )
    meta = json.loads(model_meta_str)
    party_features_name, _ = get_party_features_info(meta)

    party_used_schemas = {}
    for party, _ in input_schema.items():
        if party in party_features_name:
            used_schemas = party_features_name[party]
        else:
            used_schemas = []

        party_used_schemas[party] = {"used": set(used_schemas)}

    return party_used_schemas


def ss_sgd_converter(
    ctx,
    node_id: int,
    builder: GraphBuilderManager,
    spu_config: SPU,
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
    pred_name: str,
    traced_input: Dict[str, Set[str]],
):
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    model_objs, model_meta_str = model_loads(
        ctx,
        model_ds,
        sgd.MODEL_MAX_MAJOR_VERSION,
        sgd.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_SGD_MODEL,
        spu=spu,
    )
    meta = json.loads(model_meta_str)
    spu_w = model_objs[0]

    feature_names = meta["feature_names"]
    party_features_name, party_features_pos = get_party_features_info(meta)
    reg_type = RegType(meta["reg_type"])
    sig_type = SigType(meta["sig_type"])
    label_col = meta["label_col"]
    assert len(label_col) == 1
    label_col = label_col[0]

    if reg_type == RegType.Linear:
        link_type = LinkFunctionType.LF_IDENTITY
    else:
        link_type = SS_SGD_LINK_MAP[sig_type]

    (
        party_dot_kwargs,
        party_merge_kwargs,
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_merge_input_schemas,
        party_merge_output_schemas,
    ) = linear_model_converter(
        party_features_name,
        party_features_pos,
        input_schema,
        feature_names,
        spu_w,
        label_col,
        None,
        1.0,
        link_type,
        pred_name,
        traced_input,
    )

    builder.add_node(
        f"ss_sgd_{node_id}_dot",
        "dot_product",
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_dot_kwargs,
    )
    builder.new_execution("DP_ANYONE")
    builder.add_node(
        f"ss_sgd_{node_id}_merge_y",
        "merge_y",
        party_merge_input_schemas,
        party_merge_output_schemas,
        party_merge_kwargs,
    )


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

    for pos in range(len(leaf_node_ids)):
        lchild_ids.append(-1)
        rchild_ids.append(-1)

    node_ids.extend(leaf_node_ids)
    assert (
        len(node_ids) == len(lchild_ids)
        and len(node_ids) == len(rchild_ids)
        and len(node_ids) == len(split_feature_indices)
        and len(node_ids) == len(split_values)
    ), f"len of node_ids lchild_ids rchild_ids leaf_flags split_feature_indices split_values mismatch, {len(node_ids)} vs {len(lchild_ids)} vs {len(rchild_ids)} vs {len(leaf_flags)} vs {len(split_feature_indices)} vs {len(split_values)}"

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


def ss_xgb_schema_info(
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
):
    model_meta_str = model_meta_info(
        model_ds,
        xgb.MODEL_MAX_MAJOR_VERSION,
        xgb.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_XGB_MODEL,
    )
    meta = json.loads(model_meta_str)

    party_features_name, _ = get_party_features_info(meta)

    party_used_schemas = {}
    for party in input_schema:
        if party in party_features_name:
            used_schemas = party_features_name[party]
        else:
            used_schemas = []

        party_used_schemas[party] = {"used": set(used_schemas)}

    return party_used_schemas


def ss_xgb_converter(
    ctx,
    node_id: int,
    builder: GraphBuilderManager,
    spu_config,
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
    pred_name: str,
    traced_input: Dict[str, Set[str]],
):
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    pyus = {p: PYU(p) for p in ctx.cluster_config.desc.parties}

    model_objs, model_meta_str = model_loads(
        ctx,
        model_ds,
        xgb.MODEL_MAX_MAJOR_VERSION,
        xgb.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_XGB_MODEL,
        pyus,
        spu=spu,
    )
    model_meta = json.loads(model_meta_str)
    model = build_ss_xgb_model(model_objs, model_meta_str, spu)

    party_features_name, _ = get_party_features_info(model_meta)
    tree_num = model_meta["tree_num"]
    label_col = model_meta["label_col"]
    assert len(label_col) == 1
    label_col = label_col[0]

    assert set(party_features_name).issubset(set(input_schema))
    assert len(party_features_name) > 0

    if model.get_objective() == SSXgbRegType.Logistic:
        # refer to `XgbModel.predict`
        algo_func_type = LinkFunctionType.LF_SIGMOID_SR
    else:
        algo_func_type = LinkFunctionType.LF_IDENTITY
    algo_func = LinkFunctionType.Name(algo_func_type)

    trees = model.get_trees()
    spu_ws = model.get_weights()

    select_parent_node = builder.get_last_node_name()

    party_specific_flag = {}
    all_merge_kwargs = []
    tree_select_names = []

    party_predict_inputs = dict()
    party_predict_outputs = dict()
    party_predict_kwargs = dict()
    for tree_pos in range(tree_num):
        tree_dict = trees[tree_pos]
        spu_w = spu_ws[tree_pos]

        party_select_kwargs = dict()
        party_select_inputs = dict()
        party_select_outputs = dict()
        party_merge_kwargs = dict()
        party_merge_inputs = dict()
        party_merge_outputs = dict()
        for party, input_features in input_schema.items():
            assert party in input_schema
            pyu = pyus[party]
            assert party in traced_input

            if party in party_features_name:
                party_features = party_features_name[party]
                assert set(party_features).issubset(set(input_features))

                tree = tree_dict[pyu]

                # refer to `hnp.tree_predict`
                def build_xgb_tree_attrs(tree: XgbTree):
                    node_ids = [i for i in range(len(tree.split_features))]
                    split_feature_indices = tree.split_features
                    split_values = tree.split_values

                    return build_tree_attrs(
                        node_ids, split_feature_indices, split_values
                    )

                pyu_tree_attr_list = pyu(build_xgb_tree_attrs, num_returns=6)(tree)
            else:
                party_features = []
                pyu_tree_attr_list = [[]] * 6

            pyu_node_ids = pyu_tree_attr_list[0]
            pyu_lchild_ids = pyu_tree_attr_list[1]
            pyu_rchild_ids = pyu_tree_attr_list[2]
            pyu_split_feature_indices = pyu_tree_attr_list[3]
            pyu_split_values = pyu_tree_attr_list[4]
            pyu_leaf_node_ids = pyu_tree_attr_list[5]

            assert set(party_features).issubset(traced_input[party])

            party_select_inputs[pyu] = Table.from_schema(
                {f: input_features[f] for f in party_features}
            ).dump_serving_pb("tmp")[1]
            party_select_outputs[pyu] = Table.from_schema(
                {"selects": np.float32}
            ).dump_serving_pb("tmp")[1]

            party_select_kwargs[pyu] = {
                "input_feature_names": party_features,
                "input_feature_types": [
                    sf_type_to_serving_str(input_features[f]) for f in party_features
                ],
                "output_col_name": "selects",
                "root_node_id": 0,
                "node_ids": pyu_node_ids,
                "lchild_ids": pyu_lchild_ids,
                "rchild_ids": pyu_rchild_ids,
                "split_feature_idxs": pyu_split_feature_indices,
                "split_values": pyu_split_values,
                "leaf_node_ids": pyu_leaf_node_ids,
            }

            party_merge_inputs[pyu] = Table.from_schema(
                {"selects": np.float32}
            ).dump_serving_pb("tmp")[1]
            party_merge_outputs[pyu] = Table.from_schema(
                {"weights": np.float32}
            ).dump_serving_pb("tmp")[1]

            party_merge_kwargs[pyu] = {
                "input_col_name": "selects",
                "output_col_name": "weights",
            }
            party_specific_flag[pyu] = False
            if label_col in input_features:
                pyu_w = pyu(lambda w: list(w))(spu_w.to(pyu))
                party_specific_flag[pyu] = True
                party_merge_kwargs[pyu]["leaf_weights"] = pyu_w

        all_merge_kwargs.append(party_merge_kwargs)
        select_node_name = f"xgb_{node_id}_select_{tree_pos}"
        builder.add_node(
            select_node_name,
            "tree_select",
            party_select_inputs,
            party_select_outputs,
            party_select_kwargs,
            [select_parent_node] if select_parent_node else [],
        )
        tree_select_names.append(select_node_name)

    for party in input_schema.keys():
        party_predict_inputs[pyus[party]] = Table.from_schema(
            {"weights": np.float32}
        ).dump_serving_pb("tmp")[1]
        party_predict_outputs[pyus[party]] = Table.from_schema(
            {pred_name: np.float32}
        ).dump_serving_pb("tmp")[1]
        party_predict_kwargs[pyus[party]] = {
            "input_col_name": "weights",
            "output_col_name": pred_name,
            "algo_func": algo_func,
            "num_trees": tree_num,
        }

    builder.new_execution("DP_SPECIFIED", party_specific_flag=party_specific_flag)

    # add tree_merge nodes
    predict_parent_node_names = []
    for pos, party_merge_kwargs in enumerate(all_merge_kwargs):
        n_name = f"xgb_{node_id}_merge_{pos}"
        parents = [tree_select_names[pos]]
        assert len(parents) == 1
        builder.add_node(
            n_name,
            "tree_merge",
            party_merge_inputs,
            party_merge_outputs,
            party_merge_kwargs,
            parents,
        )
        predict_parent_node_names.append(n_name)

    # add tree_ensemble_predict node
    builder.add_node(
        f"xgb_{node_id}_predict",
        "tree_ensemble_predict",
        party_predict_inputs,
        party_predict_outputs,
        party_predict_kwargs,
        predict_parent_node_names,
    )


def sgb_schema_info(
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
):
    model_meta_str = model_meta_info(
        model_ds,
        sgb.MODEL_MAX_MAJOR_VERSION,
        sgb.MODEL_MAX_MINOR_VERSION,
        DistDataType.SGB_MODEL,
    )
    meta = json.loads(model_meta_str)
    party_features_name, _ = get_party_features_info(meta)

    party_used_schemas = {}
    for party in input_schema:
        if party in party_features_name:
            used_schemas = party_features_name[party]
        else:
            used_schemas = []

        party_used_schemas[party] = {"used": set(used_schemas)}

    return party_used_schemas


def sgb_converter(
    ctx,
    node_id: int,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
    pred_name: str,
    traced_input: Dict[str, Set[str]],
):
    pyus = {p: PYU(p) for p in ctx.cluster_config.desc.parties}

    model_objs, model_meta_str = model_loads(
        ctx,
        model_ds,
        sgb.MODEL_MAX_MAJOR_VERSION,
        sgb.MODEL_MAX_MINOR_VERSION,
        DistDataType.SGB_MODEL,
        pyus,
    )
    model_meta = json.loads(model_meta_str)

    sgb_model = sgb.build_sgb_model(pyus, model_objs, model_meta_str)

    party_features_name, _ = get_party_features_info(model_meta)

    tree_num = model_meta["common"]["tree_num"]
    label_col = model_meta["label_col"]
    assert len(label_col) == 1
    label_col = label_col[0]

    assert set(party_features_name).issubset(set(input_schema))
    assert len(party_features_name) > 0

    if sgb_model.get_objective() == SgbRegType.Logistic:
        # refer to `SgbModel.predict`
        algo_func_type = LinkFunctionType.LF_SIGMOID_SR
    else:
        algo_func_type = LinkFunctionType.LF_IDENTITY
    algo_func = LinkFunctionType.Name(algo_func_type)

    select_parent_node = builder.get_last_node_name()

    dist_trees = sgb_model.get_trees()

    party_specific_flag = {}
    all_merge_kwargs = []
    tree_select_names = []
    party_predict_inputs = dict()
    party_predict_outputs = dict()
    party_predict_kwargs = dict()
    for tree_pos in range(tree_num):
        dist_tree = dist_trees[tree_pos]
        split_tree_dict = dist_tree.get_split_tree_dict()

        party_select_kwargs = dict()
        party_select_inputs = dict()
        party_select_outputs = dict()
        party_merge_kwargs = dict()
        party_merge_inputs = dict()
        party_merge_outputs = dict()
        for party, input_features in input_schema.items():
            pyu = pyus[party]
            assert party in traced_input

            if party in party_features_name:
                party_features = party_features_name[party]
                assert set(party_features).issubset(set(input_features))

                # split tree
                assert pyu in split_tree_dict
                split_tree = split_tree_dict[pyu]

                # refer to `hnp.tree_predict_with_indices`
                def build_sgb_tree_attrs(tree):
                    node_ids = tree.split_indices
                    split_feature_indices = tree.split_features
                    split_values = tree.split_values

                    return build_tree_attrs(
                        node_ids, split_feature_indices, split_values
                    )

                pyu_tree_attr_list = pyu(build_sgb_tree_attrs, num_returns=6)(
                    split_tree
                )
            else:
                party_features = []
                pyu_tree_attr_list = [[]] * 6

            pyu_node_ids = pyu_tree_attr_list[0]
            pyu_lchild_ids = pyu_tree_attr_list[1]
            pyu_rchild_ids = pyu_tree_attr_list[2]
            pyu_split_feature_indices = pyu_tree_attr_list[3]
            pyu_split_values = pyu_tree_attr_list[4]
            pyu_leaf_node_ids = pyu_tree_attr_list[5]

            assert set(party_features).issubset(traced_input[party])

            party_select_inputs[pyu] = Table.from_schema(
                {f: input_features[f] for f in party_features}
            ).dump_serving_pb("tmp")[1]
            party_select_outputs[pyu] = Table.from_schema(
                {"selects": np.float32}
            ).dump_serving_pb("tmp")[1]

            party_select_kwargs[pyu] = {
                "input_feature_names": party_features,
                "input_feature_types": [
                    sf_type_to_serving_str(input_features[f]) for f in party_features
                ],
                "output_col_name": "selects",
                "root_node_id": 0,
                "node_ids": pyu_node_ids,
                "lchild_ids": pyu_lchild_ids,
                "rchild_ids": pyu_rchild_ids,
                "split_feature_idxs": pyu_split_feature_indices,
                "split_values": pyu_split_values,
                "leaf_node_ids": pyu_leaf_node_ids,
            }

            party_merge_inputs[pyu] = Table.from_schema(
                {"selects": np.float32}
            ).dump_serving_pb("tmp")[1]
            party_merge_outputs[pyu] = Table.from_schema(
                {"weights": np.float32}
            ).dump_serving_pb("tmp")[1]

            party_merge_kwargs[pyu] = {
                "input_col_name": "selects",
                "output_col_name": "weights",
            }
            party_specific_flag[pyu] = False
            if label_col in input_features:
                party_specific_flag[pyu] = True
                party_merge_kwargs[pyu]["leaf_weights"] = pyu(
                    lambda weight: list(weight)
                )(dist_tree.get_leaf_weight())

        all_merge_kwargs.append(party_merge_kwargs)
        select_node_name = f"sgb_{node_id}_select_{tree_pos}"
        builder.add_node(
            select_node_name,
            "tree_select",
            party_select_inputs,
            party_select_outputs,
            party_select_kwargs,
            [select_parent_node] if select_parent_node else [],
        )
        tree_select_names.append(select_node_name)

    for party in input_schema.keys():
        party_predict_inputs[pyus[party]] = Table.from_schema(
            {"weights": np.float32}
        ).dump_serving_pb("tmp")[1]
        party_predict_outputs[pyus[party]] = Table.from_schema(
            {pred_name: np.float32}
        ).dump_serving_pb("tmp")[1]
        party_predict_kwargs[pyus[party]] = {
            "input_col_name": "weights",
            "output_col_name": pred_name,
            "algo_func": algo_func,
            "num_trees": tree_num,
        }

    builder.new_execution("DP_SPECIFIED", party_specific_flag=party_specific_flag)

    # add tree_merge nodes
    predict_parent_node_names = []
    for pos, party_merge_kwargs in enumerate(all_merge_kwargs):
        n_name = f"sgb_{node_id}_merge_{pos}"
        parents = [tree_select_names[pos]]
        assert len(parents) == 1
        builder.add_node(
            n_name,
            "tree_merge",
            party_merge_inputs,
            party_merge_outputs,
            party_merge_kwargs,
            parents,
        )
        predict_parent_node_names.append(n_name)

    # add tree_ensemble_predict node
    builder.add_node(
        f"sgb_{node_id}_predict",
        "tree_ensemble_predict",
        party_predict_inputs,
        party_predict_outputs,
        party_predict_kwargs,
        predict_parent_node_names,
    )


class TrainModelConverter:
    def __init__(
        self,
        ctx,
        builder: GraphBuilderManager,
        node_id: int,
        spu_config,
        param: NodeEvalParam,
        in_ds: List[DistData],
        out_ds: List[DistData],
    ):
        self.ctx = ctx
        self.builder = builder
        self.node_id = node_id
        self.spu_config = spu_config
        self.param = param
        in_dataset = [d for d in in_ds if d.type == DistDataType.VERTICAL_TABLE]
        assert len(in_dataset) == 1
        self.in_dataset = in_dataset[0]

        self.input_schema = extract_table_header(
            self.in_dataset, load_features=True, load_ids=True, load_labels=True
        )

        assert set(self.input_schema) == set([p.party for p in builder.pyus])

        if param.name in ["ss_glm_predict", "ss_glm_train"]:
            # SS_GLM_MODEL
            if param.name == "ss_glm_train":
                model_ds = [d for d in out_ds if d.type == DistDataType.SS_GLM_MODEL]
            else:
                model_ds = [d for d in in_ds if d.type == DistDataType.SS_GLM_MODEL]
            assert len(model_ds) == 1
            self.model_ds = model_ds[0]
        elif param.name in ["ss_sgd_train", "ss_sgd_predict"]:
            # SS_SGD_MODEL
            if param.name == "ss_sgd_train":
                model_ds = [d for d in out_ds if d.type == DistDataType.SS_SGD_MODEL]
            else:
                model_ds = [d for d in in_ds if d.type == DistDataType.SS_SGD_MODEL]
            assert len(model_ds) == 1
            self.model_ds = model_ds[0]
        elif param.name in ["sgb_train", "sgb_predict"]:
            # SGB_MODEL
            if param.name == "sgb_train":
                model_ds = [d for d in out_ds if d.type == DistDataType.SGB_MODEL]
            else:
                model_ds = [d for d in in_ds if d.type == DistDataType.SGB_MODEL]
            assert len(model_ds) == 1
            self.model_ds = model_ds[0]
        elif param.name in ["ss_xgb_train", "ss_xgb_predict"]:
            # SGB_MODEL
            if param.name == "ss_xgb_train":
                model_ds = [d for d in out_ds if d.type == DistDataType.SS_XGB_MODEL]
            else:
                model_ds = [d for d in in_ds if d.type == DistDataType.SS_XGB_MODEL]
            assert len(model_ds) == 1
            self.model_ds = model_ds[0]
        else:
            # TODO others model
            raise AttributeError(f"not support param.name {param.name}")

    def schema_info(self) -> Dict[str, Dict[str, Set[str]]]:
        if self.param.name in ["ss_glm_predict", "ss_glm_train"]:
            # SS_GLM_MODEL
            return ss_glm_schema_info(
                self.input_schema,
                self.model_ds,
            )
        elif self.param.name in ["ss_sgd_train", "ss_sgd_predict"]:
            # SS_SGD_MODEL
            return ss_sgd_schema_info(
                self.input_schema,
                self.model_ds,
            )
        elif self.param.name in ["sgb_train", "sgb_predict"]:
            # SGB_MODEL
            return sgb_schema_info(
                self.input_schema,
                self.model_ds,
            )
        elif self.param.name in ["ss_xgb_train", "ss_xgb_predict"]:
            # SGB_MODEL
            return ss_xgb_schema_info(
                self.input_schema,
                self.model_ds,
            )
        else:
            # TODO others model
            raise AttributeError(f"not support param.name {self.param.name}")

    def convert(
        self, traced_input: Dict[str, Set[str]], traced_output: Dict[str, Set[str]]
    ):
        assert len(traced_output) == 0
        if self.param.name in ["ss_glm_predict", "ss_glm_train"]:
            # SS_GLM_MODEL
            ss_glm_converter(
                self.ctx,
                self.node_id,
                self.builder,
                self.spu_config,
                self.input_schema,
                self.model_ds,
                "pred_y",
                traced_input,
            )
        elif self.param.name in ["ss_sgd_train", "ss_sgd_predict"]:
            # SS_SGD_MODEL
            ss_sgd_converter(
                self.ctx,
                self.node_id,
                self.builder,
                self.spu_config,
                self.input_schema,
                self.model_ds,
                "pred_y",
                traced_input,
            )
        elif self.param.name in ["sgb_train", "sgb_predict"]:
            # SGB_MODEL
            sgb_converter(
                self.ctx,
                self.node_id,
                self.builder,
                self.input_schema,
                self.model_ds,
                "pred_y",
                traced_input,
            )
        elif self.param.name in ["ss_xgb_train", "ss_xgb_predict"]:
            # SGB_MODEL
            ss_xgb_converter(
                self.ctx,
                self.node_id,
                self.builder,
                self.spu_config,
                self.input_schema,
                self.model_ds,
                "pred_y",
                traced_input,
            )
        else:
            # TODO others model
            raise AttributeError(f"not support param.name {self.param.name}")
