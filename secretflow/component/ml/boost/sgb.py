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

from secretflow_serving_lib.link_function_pb2 import LinkFunctionType
from secretflow_spec.v1.data_pb2 import DistData

from secretflow.component.core import (
    SGB_MODEL_MAX,
    Context,
    DistDataType,
    ServingBuilder,
    VTable,
    VTableUtils,
)
from secretflow.device import PYU
from secretflow.ml.boost.sgb_v.checkpoint import SGBSnapshot, build_sgb_model
from secretflow.ml.boost.sgb_v.core.params import RegType

from .boost import (
    TreeNodeContext,
    build_tree_attrs,
    build_tree_model,
    get_party_features_info,
)


class SGBExportMixin:
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

        pyus = {p: PYU(p) for p in ctx.parties}
        model = ctx.load_model(
            model_dd, DistDataType.SGB_MODEL, SGB_MODEL_MAX, pyus=pyus
        )
        model_meta = json.loads(model.public_info)

        sgb_model = build_sgb_model(SGBSnapshot(model.objs, model_meta))

        party_features_name, _ = get_party_features_info(model_meta)

        tree_num = model_meta["common"]["tree_num"]
        label_col = model_meta["label_col"]
        assert len(label_col) == 1
        label_col = label_col[0]

        assert set(party_features_name).issubset(set(input_tbl.schemas))
        assert len(party_features_name) > 0

        if sgb_model.get_objective() == RegType.Logistic:
            # refer to `SgbModel.predict`
            algo_func_type = LinkFunctionType.LF_SIGMOID_RAW
        elif sgb_model.get_objective() == RegType.Tweedie:
            algo_func_type = LinkFunctionType.LF_EXP
        else:
            algo_func_type = LinkFunctionType.LF_IDENTITY
        algo_func = LinkFunctionType.Name(algo_func_type)

        select_parent_node = builder.get_last_node_name()

        dist_trees = sgb_model.get_trees()

        tree_node_ctxs = []
        for tree_pos in range(tree_num):
            dist_tree = dist_trees[tree_pos]
            split_tree_dict = dist_tree.get_split_tree_dict()

            tree_node_ctx = TreeNodeContext()
            for party, input_features in input_tbl.schemas.items():
                pyu = pyus[party]

                if party in party_features_name:
                    party_features = party_features_name[party]
                    assert set(party_features).issubset(set(input_features.names))

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

                tree_node_ctx.add_party_attr(
                    pyu,
                    input_features,
                    party_features,
                    label_col,
                    pyu_node_ids,
                    pyu_lchild_ids,
                    pyu_rchild_ids,
                    pyu_split_feature_indices,
                    pyu_split_values,
                    pyu_leaf_node_ids,
                    dist_tree.get_leaf_weight(),
                )
            tree_node_ctxs.append(tree_node_ctx)

        build_tree_model(
            tree_node_ctxs,
            builder,
            "sgb",
            node_id,
            pyus,
            input_tbl.schemas.keys(),
            pred_name,
            tree_num,
            algo_func,
            sgb_model.base,
            select_parent_node,
            he_mode,
        )
