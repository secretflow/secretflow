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


import json

import numpy as np
from secretflow_serving_lib.link_function_pb2 import LinkFunctionType

from secretflow.component.core import (
    SS_XGB_MODEL_MAX,
    Context,
    DispatchType,
    DistDataType,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
    VTable,
)
from secretflow.compute import Table
from secretflow.device import PYU
from secretflow.ml.boost.sgb_v.core.params import RegType
from secretflow.ml.boost.ss_xgb_v.checkpoint import (
    SSXGBCheckpointData,
    build_ss_xgb_model,
)
from secretflow.ml.boost.ss_xgb_v.core.node_split import RegType
from secretflow.ml.boost.ss_xgb_v.core.xgb_tree import XgbTree
from secretflow.spec.v1.data_pb2 import DistData

from .boost import build_tree_attrs, get_party_features_info


class SSXGBExportMixin:
    def do_export(
        self,
        ctx: Context,
        builder: ServingBuilder,
        input_dd: DistData,
        model_dd: DistData,
        he_mode: bool,
        pred_name: str = "pred_y",
    ):
        assert he_mode is False, "feature not supported yet. change `he_mode` to False."

        input_tbl = VTable.from_distdata(input_dd)
        node_id = builder.max_id()

        spu = ctx.make_spu()
        pyus = {p: PYU(p) for p in ctx.parties}
        model = ctx.load_model(
            model_dd, DistDataType.SS_XGB_MODEL, SS_XGB_MODEL_MAX, pyus=pyus, spu=spu
        )
        model_meta = json.loads(model.public_info)
        ss_xgb_model = build_ss_xgb_model(
            SSXGBCheckpointData(model.objs, model_meta), spu
        )

        party_features_name, _ = get_party_features_info(model_meta)
        tree_num = model_meta["tree_num"]
        label_col = model_meta["label_col"]
        assert len(label_col) == 1
        label_col = label_col[0]

        assert set(party_features_name).issubset(set(input_tbl.schemas))
        assert len(party_features_name) > 0

        if ss_xgb_model.get_objective() == RegType.Logistic:
            # refer to `XgbModel.predict`
            algo_func_type = LinkFunctionType.LF_SIGMOID_SR
        else:
            algo_func_type = LinkFunctionType.LF_IDENTITY
        algo_func = LinkFunctionType.Name(algo_func_type)

        trees = ss_xgb_model.get_trees()
        spu_ws = ss_xgb_model.get_weights()

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
            for party, input_features in input_tbl.schemas.items():
                pyu = pyus[party]
                # assert party in traced_input

                if party in party_features_name:
                    party_features = party_features_name[party]
                    assert set(party_features).issubset(set(input_features.names))

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

                party_select_inputs[pyu] = Table.from_schema(
                    input_features.select(party_features).to_arrow()
                ).dump_serving_pb("tmp")[1]
                party_select_outputs[pyu] = Table.from_schema(
                    {"selects": np.uint64}
                ).dump_serving_pb("tmp")[1]

                party_select_kwargs[pyu] = {
                    "input_feature_names": party_features,
                    "input_feature_types": [
                        input_features[f].ftype.to_serving_dtype()
                        for f in party_features
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
                    {"selects": np.uint64}
                ).dump_serving_pb("tmp")[1]
                party_merge_outputs[pyu] = Table.from_schema(
                    {"weights": np.float64}
                ).dump_serving_pb("tmp")[1]

                party_merge_kwargs[pyu] = {
                    "input_col_name": "selects",
                    "output_col_name": "weights",
                }
                party_specific_flag[pyu] = False
                if label_col in input_features.fields:
                    pyu_w = pyu(lambda w: list(w))(spu_w.to(pyu))
                    party_specific_flag[pyu] = True
                    party_merge_kwargs[pyu]["leaf_weights"] = pyu_w

            all_merge_kwargs.append(party_merge_kwargs)
            select_node_name = f"ss_xgb_{node_id}_select_{tree_pos}"
            node = ServingNode(
                select_node_name,
                ServingOp.TREE_SELECT,
                ServingPhase.TRAIN_PREDICT,
                party_select_inputs,
                party_select_outputs,
                party_select_kwargs,
                [select_parent_node] if select_parent_node else [],
            )
            builder.add_node(node)
            tree_select_names.append(select_node_name)

        for party in input_tbl.schemas.keys():
            party_predict_inputs[pyus[party]] = Table.from_schema(
                {"weights": np.float64}
            ).dump_serving_pb("tmp")[1]
            party_predict_outputs[pyus[party]] = Table.from_schema(
                {pred_name: np.float64}
            ).dump_serving_pb("tmp")[1]
            party_predict_kwargs[pyus[party]] = {
                "input_col_name": "weights",
                "output_col_name": pred_name,
                "algo_func": algo_func,
                "num_trees": tree_num,
                "base_score": ss_xgb_model.base,
            }

        builder.new_execution(
            DispatchType.DP_SPECIFIED, party_specific_flag=party_specific_flag
        )

        # add tree_merge nodes
        predict_parent_node_names = []
        for pos, party_merge_kwargs in enumerate(all_merge_kwargs):
            n_name = f"ss_xgb_{node_id}_merge_{pos}"
            parents = [tree_select_names[pos]]
            assert len(parents) == 1
            node = ServingNode(
                n_name,
                ServingOp.TREE_MERGE,
                ServingPhase.TRAIN_PREDICT,
                party_merge_inputs,
                party_merge_outputs,
                party_merge_kwargs,
                parents,
            )
            builder.add_node(node)

            predict_parent_node_names.append(n_name)

        # add tree_ensemble_predict node
        node = ServingNode(
            f"ss_xgb_{node_id}_predict",
            ServingOp.TREE_ENSEMBLE_PREDICT,
            ServingPhase.TRAIN_PREDICT,
            party_predict_inputs,
            party_predict_outputs,
            party_predict_kwargs,
            predict_parent_node_names,
        )
        builder.add_node(node)
