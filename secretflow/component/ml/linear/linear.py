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


import numpy as np
from secretflow_serving_lib.link_function_pb2 import LinkFunctionType

from secretflow.component.core import (
    DispatchType,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
    VTableSchema,
    VTableUtils,
)
from secretflow.compute import Table
from secretflow.device import HEU, PYU, SPU, PYUObject, SPUObject
from secretflow.device.device.heu import HEUMoveConfig


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


def spu_weight_slice(spu: SPU, spu_w: SPUObject, start: int, end: int):
    def _slice(w):
        w = w.reshape((-1, 1))
        return w.flatten()[start:end]

    return spu(_slice)(spu_w)


def reveal_to_pyu(
    spu: SPU, spu_w: SPUObject, start: int, end: int, to: PYU
) -> PYUObject:
    sliced_w = spu_weight_slice(spu, spu_w, start, end)
    return to(lambda w: list(w))(sliced_w.to(to))


def to_serialized_pyu(
    spu: SPU,
    spu_w: SPUObject,
    start: int,
    end: int,
    heu: HEU,
    move_config: HEUMoveConfig,
    to: PYU,
) -> PYUObject:
    sliced_w = spu_weight_slice(spu, spu_w, start, end)
    heu_w = sliced_w.to(heu, move_config)
    return heu_w.serialize_to_pyu(to)


def build_linear_model(
    builder: ServingBuilder,
    node_prefix: str,
    party_features_name: dict[str, list[str]],
    party_features_pos: dict[str, tuple[int, int]],
    input_schema: dict[str, VTableSchema],
    feature_names: list[str],
    spu_w: SPUObject,
    label_col: str,
    offset_col: str | None,
    yhat_scale: float,
    link_type: LinkFunctionType,
    exp_iters: int,
    pred_name: str = "pred_y",
) -> None:
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

        if party in party_features_name:
            party_features = party_features_name[party]
            assert set(party_features).issubset(set(input_features.names))
            start, end = party_features_pos[party]
            pyu_w = reveal_to_pyu(spu, spu_w, start, end, pyu)
        else:
            party_features = []
            pyu_w = pyu(lambda: [])()

        if offset_col and offset_col in input_features:
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

        party_dot_input_schemas[pyu] = Table.from_schema(
            VTableUtils.to_arrow_schema(input_features.select(party_features))
        ).dump_serving_pb("tmp")[1]
        party_dot_output_schemas[pyu] = Table.from_schema(
            {"partial_y": np.float64}
        ).dump_serving_pb("tmp")[1]

        party_dot_kwargs[pyu] = {
            "feature_names": party_features,
            "feature_weights": pyu_w,
            "input_types": [
                VTableUtils.to_serving_dtype(input_features[f].type)
                for f in party_features
            ],
            "output_col_name": "partial_y",
            "intercept": intercept,
        }

        party_merge_input_schemas[pyu] = Table.from_schema(
            {"partial_y": np.float64}
        ).dump_serving_pb("tmp")[1]
        party_merge_output_schemas[pyu] = Table.from_schema(
            {pred_name: np.float64}
        ).dump_serving_pb("tmp")[1]

        party_merge_kwargs[pyu] = {
            "yhat_scale": yhat_scale,
            "link_function": LinkFunctionType.Name(link_type),
            "exp_iters": exp_iters,
            "input_col_name": "partial_y",
            "output_col_name": pred_name,
        }

    node_id = builder.max_id()
    node = ServingNode(
        f"{node_prefix}_{node_id}_dot",
        ServingOp.DOT_PRODUCT,
        ServingPhase.TRAIN_PREDICT,
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_dot_kwargs,
    )
    builder.add_node(node)
    builder.new_execution(DispatchType.DP_ANYONE)
    node = ServingNode(
        f"{node_prefix}_{node_id}_merge_y",
        ServingOp.MERGE_Y,
        ServingPhase.TRAIN_PREDICT,
        party_merge_input_schemas,
        party_merge_output_schemas,
        party_merge_kwargs,
    )
    builder.add_node(node)


def build_phe_linear_model(
    builder: ServingBuilder,
    node_prefix: str,
    heu_dict: dict[str, HEU],
    party_features_name: dict[str, list[str]],
    party_features_pos: dict[str, tuple[int, int]],
    input_schema: dict[str, VTableSchema],
    feature_names: list[str],
    spu_w: SPUObject,
    label_col: str,
    offset_col: str,
    yhat_scale: float,
    link_type: LinkFunctionType,
    exp_iters: int,
    pred_name: str = "pred_y",
):
    assert set(party_features_name).issubset(set(input_schema))
    assert set(party_features_name) == set(party_features_pos)
    assert len(party_features_name) <= 2

    peer_parties = {}
    parties = list(input_schema.keys())
    peer_parties[parties[0]] = parties[1]
    peer_parties[parties[1]] = parties[0]

    spu = spu_w.device
    party_dot_input_schemas = dict()
    party_dot_output_schemas = dict()
    party_dot_kwargs = dict()

    party_reduce_self_kwargs = dict()
    party_reduce_self_input_schemas = dict()
    party_reduce_self_output_schemas = dict()
    party_reduce_peer_kwargs = dict()
    party_reduce_peer_input_schemas = dict()
    party_reduce_peer_output_schemas = dict()

    party_decrypt_kwargs = dict()
    party_decrypt_input_schemas = dict()
    party_decrypt_output_schemas = dict()

    party_merge_kwargs = dict()
    party_merge_input_schemas = dict()
    party_merge_output_schemas = dict()

    for party, input_features in input_schema.items():
        pyu = PYU(party)

        party_dot_kwargs[pyu] = {}

        peer_heu = heu_dict[peer_parties[party]]
        move_config = HEUMoveConfig(
            heu_dest_party=party,
            heu_encoder=peer_heu.encoder,
        )

        if party in party_features_name:
            party_features = party_features_name[party]
            assert set(party_features).issubset(set(input_features.names))
            start, end = party_features_pos[party]
            pyu_w = to_serialized_pyu(
                spu, spu_w, start, end, peer_heu, move_config, pyu
            )
        else:
            party_features = []
            pyu_w = None

        if offset_col in input_features:
            party_features.append(offset_col)
            party_dot_kwargs[pyu]["offset_col_name"] = offset_col

        if label_col in input_features:
            intercept = to_serialized_pyu(
                spu,
                spu_w,
                len(feature_names),
                len(feature_names) + 1,
                peer_heu,
                move_config,
                pyu,
            )
            party_dot_kwargs[pyu]["intercept_ciphertext"] = intercept

        # dot product op
        if len(party_features) > 0:
            party_dot_kwargs[pyu]["feature_names"] = party_features
            party_dot_kwargs[pyu]["feature_types"] = [
                VTableUtils.to_serving_dtype(input_features[f].type)
                for f in party_features
            ]
            if pyu_w is not None:
                party_dot_kwargs[pyu]["feature_weights_ciphertext"] = pyu_w

        party_dot_kwargs[pyu]["result_col_name"] = "partial_y"
        party_dot_kwargs[pyu]["rand_number_col_name"] = "rand"
        party_dot_input_schemas[pyu] = Table.from_schema(
            VTableUtils.to_arrow_schema(input_features.select(party_features))
        ).dump_serving_pb("tmp")[1]
        party_dot_output_schemas[pyu] = Table.from_schema(
            {"partial_y": np.bytes_, "rand:": np.bytes_}
        ).dump_serving_pb("tmp")[1]

        # reduct op
        party_reduce_self_kwargs[pyu] = {
            "partial_y_col_name": "partial_y",
            "rand_number_col_name": "rand",
            "select_crypted_for_peer": False,
        }
        party_reduce_self_input_schemas[pyu] = Table.from_schema(
            {"partial_y": np.bytes_, "rand:": np.bytes_}
        ).dump_serving_pb("tmp")[1]
        party_reduce_self_output_schemas[pyu] = Table.from_schema(
            {"partial_y": np.bytes_}
        ).dump_serving_pb("tmp")[1]

        party_reduce_peer_kwargs[pyu] = {
            "partial_y_col_name": "partial_y",
            "rand_number_col_name": "rand",
            "select_crypted_for_peer": True,
        }
        party_reduce_peer_input_schemas[pyu] = Table.from_schema(
            {"partial_y": np.bytes_, "rand:": np.bytes_}
        ).dump_serving_pb("tmp")[1]
        party_reduce_peer_output_schemas[pyu] = Table.from_schema(
            {"partial_y": np.bytes_}
        ).dump_serving_pb("tmp")[1]

        party_decrypt_kwargs[pyu] = {
            "partial_y_col_name": party_reduce_peer_kwargs[pyu]["partial_y_col_name"],
            "decrypted_col_name": "decrypted_y",
        }
        party_decrypt_input_schemas[pyu] = party_reduce_peer_output_schemas[pyu]
        party_decrypt_output_schemas[pyu] = Table.from_schema(
            {"decrypted_y": np.bytes_}
        ).dump_serving_pb("tmp")[1]

        party_merge_kwargs[pyu] = {
            "yhat_scale": yhat_scale,
            "link_function": LinkFunctionType.Name(link_type),
            "exp_iters": exp_iters,
            "decrypted_y_col_name": party_decrypt_kwargs[pyu]["decrypted_col_name"],
            "crypted_y_col_name": party_reduce_self_kwargs[pyu]["partial_y_col_name"],
            "score_col_name": pred_name,
        }
        party_merge_input_schemas[pyu] = [
            party_decrypt_output_schemas[pyu],
            party_reduce_self_output_schemas[pyu],
        ]
        party_merge_output_schemas[pyu] = Table.from_schema(
            {party_merge_kwargs[pyu]["score_col_name"]: np.float64}
        ).dump_serving_pb("tmp")[1]

    phe_dot_node_name = f"{node_prefix}_phe_dot"
    node = ServingNode(
        phe_dot_node_name,
        ServingOp.PHE_2P_DOT_PRODUCT,
        ServingPhase.TRAIN_PREDICT,
        party_dot_input_schemas,
        party_dot_output_schemas,
        party_dot_kwargs,
    )
    builder.add_node(node)

    builder.new_execution(DispatchType.DP_SELF)
    phe_reduce_peer_node_name = f"{node_prefix}_phe_reduce_peer"
    node = ServingNode(
        phe_reduce_peer_node_name,
        ServingOp.PHE_2P_REDUCE,
        ServingPhase.TRAIN_PREDICT,
        party_reduce_peer_input_schemas,
        party_reduce_peer_output_schemas,
        party_reduce_peer_kwargs,
        [phe_dot_node_name],
    )
    builder.add_node(node)

    phe_reduce_self_node_name = f"{node_prefix}_phe_reduce_self"
    node = ServingNode(
        phe_reduce_self_node_name,
        ServingOp.PHE_2P_REDUCE,
        ServingPhase.TRAIN_PREDICT,
        party_reduce_self_input_schemas,
        party_reduce_self_output_schemas,
        party_reduce_self_kwargs,
        [phe_dot_node_name],
    )
    builder.add_node(node)

    builder.new_execution(DispatchType.DP_PEER)

    phe_decrypt_node_name = f"{node_prefix}_phe_decrypt"
    node = ServingNode(
        phe_decrypt_node_name,
        ServingOp.PHE_2P_DECRYPT_PEER_Y,
        ServingPhase.TRAIN_PREDICT,
        party_decrypt_input_schemas,
        party_decrypt_output_schemas,
        party_decrypt_kwargs,
        [phe_reduce_peer_node_name],
    )
    builder.add_node(node)

    builder.new_execution(DispatchType.DP_SELF)
    node = ServingNode(
        f"{node_prefix}_phe_merge",
        ServingOp.PHE_2P_MERGE_Y,
        ServingPhase.TRAIN_PREDICT,
        party_merge_input_schemas,
        party_merge_output_schemas,
        party_merge_kwargs,
        [phe_decrypt_node_name, phe_reduce_self_node_name],
    )
    builder.add_node(node)
