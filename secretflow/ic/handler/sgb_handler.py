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

from typing import Tuple, List
import secretflow as sf
import spu
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb, SgbModel
from google.protobuf import any_pb2
from secretflow.ic.handler.protocol_family import phe
from secretflow.ic.handler.algo import xgb
from secretflow.ic.proto.common import header_pb2
from secretflow.ic.proto.handshake import entry_pb2
from secretflow.ic.proto.handshake.algos import sgb_pb2
from secretflow.ic.proto.handshake.protocol_family import phe_pb2
from secretflow.ic.proxy import LinkProxy
from secretflow.ic.handler.handler import IcHandler
from secretflow.ic.handler import util


class SgbIcHandler(IcHandler):
    def __init__(self, config: dict, dataset: dict):
        super().__init__(dataset)
        self._xgb = xgb.XgbConfig(config['xgb'])
        self._phe = phe.PheConfig(config['heu'])

    def _build_handshake_request(self) -> entry_pb2.HandshakeRequest:
        request = entry_pb2.HandshakeRequest()

        request.version = 2

        # build sgb param
        sgb_proposal = sgb_pb2.SgbParamsProposal()
        sgb_proposal.supported_versions.append(1)
        sgb_proposal.support_completely_sgb = self._xgb.support_completely_sgb
        sgb_proposal.support_row_sample_by_tree = self._xgb.support_row_sample_by_tree
        sgb_proposal.support_col_sample_by_tree = self._xgb.support_col_sample_by_tree

        # set sgb param
        request.supported_algos.append(entry_pb2.ALGO_TYPE_SGB)
        algo_any = any_pb2.Any()
        algo_any.Pack(sgb_proposal)
        request.algo_params.append(algo_any)

        # build phe param
        phe_proposal = phe_pb2.PheProtocolProposal()
        phe_proposal.supported_versions.append(1)
        # currently only support paillier
        phe_proposal.supported_phe_algos.append(phe_pb2.PHE_ALGO_PAILLIER)
        paillier_proposal = phe_pb2.PaillierParamsProposal()
        paillier_proposal.key_sizes.extend([1024, 2048, 3072])
        param_any = any_pb2.Any()
        param_any.Pack(paillier_proposal)
        phe_proposal.supported_phe_params.append(param_any)

        # set phe param
        request.protocol_families.append(entry_pb2.PROTOCOL_FAMILY_PHE)
        protocol_any = any_pb2.Any()
        protocol_any.Pack(phe_proposal)
        request.protocol_family_params.append(protocol_any)

        return request

    def _build_handshake_response(self) -> entry_pb2.HandshakeResponse:
        response = super()._build_handshake_response()

        # build sgb param
        sgb_result = sgb_pb2.SgbParamsResult()
        sgb_result.version = 1
        sgb_result.num_round = self._xgb.num_round
        sgb_result.max_depth = self._xgb.max_depth
        sgb_result.row_sample_by_tree = self._xgb.row_sample_by_tree
        sgb_result.col_sample_by_tree = self._xgb.col_sample_by_tree
        sgb_result.bucket_eps = self._xgb.bucket_eps
        sgb_result.use_completely_sgb = self._xgb.use_completely_sgb

        # set sgb param
        response.algo = entry_pb2.ALGO_TYPE_SGB
        response.algo_param.Pack(sgb_result)

        # build phe param
        phe_result = phe_pb2.PheProtocolResult()
        phe_result.version = 1
        phe_result.phe_algo = self._phe.phe_algo

        if phe_result.phe_algo == phe_pb2.PHE_ALGO_PAILLIER:
            paillier_result = phe_pb2.PaillierParamsResult()
            paillier_result.key_size = self._phe.key_size
            phe_result.phe_param.Pack(paillier_result)

            # set phe param
            response.protocol_families.append(entry_pb2.PROTOCOL_FAMILY_PHE)
            protocol_any = any_pb2.Any()
            protocol_any.Pack(phe_result)
            response.protocol_family_params.append(protocol_any)
        else:
            raise 'unexpected behavior'

        return response

    def _process_handshake_response(
        self, response: entry_pb2.HandshakeResponse
    ) -> bool:
        if not super()._process_handshake_response(response):
            return False

        # process sgb params
        sgb_result = sgb_pb2.SgbParamsResult()
        assert entry_pb2.ALGO_TYPE_SGB == response.algo
        assert response.algo_param.Unpack(sgb_result)

        self._xgb.num_round = sgb_result.num_round
        self._xgb.max_depth = sgb_result.max_depth

        self._xgb.row_sample_by_tree = sgb_result.row_sample_by_tree
        if not self._xgb.support_row_sample_by_tree:
            assert util.almost_one(sgb_result.row_sample_by_tree)

        self._xgb.col_sample_by_tree = sgb_result.col_sample_by_tree
        if not self._xgb.support_col_sample_by_tree:
            assert util.almost_one(sgb_result.col_sample_by_tree)

        self._xgb.bucket_eps = sgb_result.bucket_eps

        self._xgb.use_completely_sgb = sgb_result.use_completely_sgb
        if not self._xgb.support_completely_sgb:
            assert not sgb_result.use_completely_sgb

        # process phe params
        phe_result = phe_pb2.PheProtocolResult()
        assert response.protocol_families[0] == entry_pb2.PROTOCOL_FAMILY_PHE
        assert response.protocol_family_params[0].Unpack(phe_result)

        self._phe.phe_algo = phe_result.phe_algo

        return True

    def _negotiate_handshake_params(
        self, requests: List[entry_pb2.HandshakeRequest]
    ) -> Tuple[int, str]:
        code, msg = self._negotiate_sgb_algo_params(requests)
        if code != header_pb2.OK:
            return code, msg

        code, msg = self._negotiate_phe_algo_params(requests)
        if code != header_pb2.OK:
            return code, msg

        return header_pb2.OK, ''

    def _negotiate_sgb_algo_params(
        self, requests: List[entry_pb2.HandshakeRequest]
    ) -> Tuple[int, str]:
        sgb_params = util.extract_req_algo_params(
            requests, entry_pb2.ALGO_TYPE_SGB, sgb_pb2.SgbParamsProposal
        )

        if sgb_params is None:
            return header_pb2.HANDSHAKE_REFUSED, 'negotiate sgb algo failed'

        if not self._negotiate_completely_sgb(sgb_params):
            return header_pb2.HANDSHAKE_REFUSED, 'negotiate completely_sgb failed'

        self._negotiate_row_sample_by_tree(sgb_params)

        self._negotiate_col_sample_by_tree(sgb_params)

        return header_pb2.OK, ''

    def _negotiate_completely_sgb(
        self, sgb_params: List[sgb_pb2.SgbParamsProposal]
    ) -> bool:
        if self._xgb.use_completely_sgb:
            support = util.align_param_item(sgb_params, 'support_completely_sgb')
            support = False if support is None else support
            if not support:
                return False

        return True

    def _negotiate_row_sample_by_tree(
        self, sgb_params: List[sgb_pb2.SgbParamsProposal]
    ):
        support = util.align_param_item(sgb_params, 'support_row_sample_by_tree')
        support = False if support is None else support
        if not support:
            self._xgb.row_sample_by_tree = 1

    def _negotiate_col_sample_by_tree(
        self, sgb_params: List[sgb_pb2.SgbParamsProposal]
    ):
        support = util.align_param_item(sgb_params, 'support_col_sample_by_tree')
        support = False if support is None else support
        if not support:
            self._xgb.col_sample_by_tree = 1

    def _negotiate_phe_algo_params(
        self, requests: List[entry_pb2.HandshakeRequest]
    ) -> Tuple[int, str]:
        phe_params = util.extract_req_protocol_family_params(
            requests, entry_pb2.PROTOCOL_FAMILY_PHE, phe_pb2.PheProtocolProposal
        )

        if phe_params is None:
            return header_pb2.HANDSHAKE_REFUSED, 'negotiate phe algo failed'

        if self._phe.phe_algo == phe_pb2.PHE_ALGO_PAILLIER:
            if not self._negotiate_paillier_params(phe_params):
                return header_pb2.HANDSHAKE_REFUSED, 'negotiate paillier params failed'
        else:
            raise f'unsupported phe algo: {self._phe.phe_algo}'

        return header_pb2.OK, ''

    def _negotiate_paillier_params(
        self, phe_params: List[phe_pb2.PheProtocolProposal]
    ) -> bool:
        paillier_params = util.extract_req_phe_params(
            phe_params, phe_pb2.PHE_ALGO_PAILLIER, phe_pb2.PaillierParamsProposal
        )

        if paillier_params is None:
            return False

        key_sizes = util.intersect_param_items(paillier_params, 'key_sizes')
        return self._phe.key_size in key_sizes

    def _run_algo(self):
        print('+++++++++++++++ run sgb ++++++++++++++++++')
        params = self._process_params()
        x, y = self._process_dataset()
        model = self._train(params, x, y)
        self._evaluate(model, x, y)

    def _process_params(self) -> dict:
        self_party = LinkProxy.self_party
        active_party = LinkProxy.all_parties[0]

        params = {
            "enable_packbits": True,
            "batch_encoding_enabled": False,
            "tree_growing_method": "level",
            "num_boost_round": self._xgb.num_round,
            "max_depth": self._xgb.max_depth,
            "sketch_eps": self._xgb.bucket_eps,
            "rowsample_by_tree": self._xgb.row_sample_by_tree,
            "colsample_by_tree": self._xgb.col_sample_by_tree,
            "first_tree_with_label_holder_feature": self._xgb.use_completely_sgb,
        }

        if self_party == active_party:
            params.update(
                {
                    "objective": self._xgb.objective,
                    "reg_lambda": self._xgb.reg_lambda,
                    "gamma": self._xgb.gamma,
                }
            )

        return params

    def _process_dataset(self) -> Tuple[FedNdarray, FedNdarray]:
        self_party = LinkProxy.self_party
        active_party = LinkProxy.all_parties[0]

        v_data = FedNdarray({}, partition_way=PartitionWay.VERTICAL)
        for party, feature in self._dataset['features'].items():
            if party == self_party:
                assert feature is not None
            party_pyu = sf.PYU(party)
            v_data.partitions.update({party_pyu: party_pyu(lambda: feature)()})

        assert active_party in self._dataset['label']
        y = self._dataset['label'][active_party]
        if self_party == active_party:
            assert y is not None
        party_pyu = sf.PYU(active_party)
        label_data = FedNdarray(
            {
                party_pyu: party_pyu(lambda: y)(),
            },
            partition_way=PartitionWay.VERTICAL,
        )

        return v_data, label_data

    def _train(self, params: dict, x: FedNdarray, y: FedNdarray) -> SgbModel:
        heu = sf.HEU(self._phe.config, spu.spu_pb2.FM128)
        sgb = Sgb(heu)
        return sgb.train(params, x, y)

    @staticmethod
    def _evaluate(model: SgbModel, x: FedNdarray, y: FedNdarray):
        print('+++++++++++++++ evaluate ++++++++++++++++++')
        yhat = model.predict(x)

        yhat = reveal(yhat)
        y = reveal(list(y.partitions.values())[0])

        from sklearn.metrics import roc_auc_score

        print(f"auc: {roc_auc_score(y, yhat)}")
