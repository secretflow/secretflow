# Copyright 2023 Ant Group Co., Ltd.
# Copyright 2023 Tsing Jiao Information Science
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

# Please read the https://github.com/secretflow/interconnection-impl/tree/main/router/README.md for the details.

import logging
from typing import List

import numpy as np
from anyconn_core import AppConfig
from heu import phe
from router import RC, Pack

from secretflow.device import PYUObject, proxy
from secretflow.utils import secure_pickle as pickle

router_table = {
    "rs_01": {"rs": ["rs_02"], "rc": ["rc_01", "rc_03"]},
    "rs_02": {"rs": ["rs_01"], "rc": ["rc_02", "rc_03"]},
}

# FIXME, you should configurate the address of rs and rc.
nodes = [
    {"id": "rs_01", "tag": "RS", "address": "101.200.76.152:30565"},
    {"id": "rs_02", "tag": "RS", "address": "101.200.76.152:30281"},
    {"id": "rc_01", "tag": "RC01", "address": "101.200.76.152:30004"},
    {"id": "rc_02", "tag": "RC02", "address": "101.200.76.152:30292"},
    {"id": "rc_03", "tag": "RC03", "address": "192.168.113.20:50063"},
]

# local test
# nodes = [
#     {"id": "rs_01", "tag": "RS", "address": "localhost:50051"},
#     {"id": "rs_02", "tag": "RS", "address": "localhost:50052"},
#     {"id": "rc_01", "tag": "RC01", "address": "localhost:50061"},
#     {"id": "rc_02", "tag": "RC02", "address": "localhost:50062"},
#     {"id": "rc_03", "tag": "RC03", "address": "localhost:50063"},
# ]


@proxy(PYUObject)
class WeightArbiter:
    def __init__(self):
        self.kit = phe.setup(phe.SchemaType.ZPaillier, 2048)
        # You can get the public key by following method:
        logging.warning(
            "\033[33m ================== Initialize Arbiter RC ===================\033[0m"
        )
        self.public_key = self.kit.public_key()
        self.rc = RC()
        self.rc_id = "rc_03"
        self.rc.run_in_thread_forever(AppConfig(node_id=self.rc_id, nodes=nodes))
        self.router_table = router_table
        self.round = 0

    def sync_with_rs(self, flatten_weights: List[float]):
        logging.warning(
            f"\033[33m ================== ROUND {self.round} ===================\033[0m"
        )
        # secretflow sends its own weights to secret-sharing platform
        w_int = [int(_w * 2**40) for _w in flatten_weights]  # left shift 40 bits.
        w2_int = [int(_w * 2**40) for _w in np.random.random(size=len(w_int))]
        w1_int = [_w - _w2 for _w, _w2 in zip(w_int, w2_int)]
        w1_pack = Pack(
            task_id="1",
            source_id=self.rc_id,
            data_id="2",
            target_id="rs_01",
            encryption="ss",
            shape=(len(w1_int),),
            dtype="int",
            data=w1_int,
            process="he2ss",
            key=None,
            n_batches=1,
            uid=f"round{self.round}__2",
            router_table=self.router_table,
        )
        w2_pack = Pack(
            task_id="1",
            source_id=self.rc_id,
            data_id="2",
            target_id="rs_02",
            encryption="ss",
            shape=(len(w2_int),),
            dtype="int",
            data=w2_int,
            process="he2ss",
            key=None,
            n_batches=1,
            uid=f"round{self.round}_2",
            router_table=self.router_table,
        )
        self.rc.send(w1_pack)
        logging.warning(f"\033[33m ROUND {self.round} rc send w1_int:{w1_int} \033[0m")
        self.rc.send(w2_pack)
        logging.warning(f"\033[33m ROUND {self.round} rc send w2_int:{w2_int} \033[0m")

        # secretflow receives aggregated global weights from secret-sharing platform.
        pk_buffer = pickle.dumps(self.public_key).decode("latin1")
        pubkey_pack = Pack(
            task_id="1",
            source_id=self.rc_id,
            data_id="2",
            target_id="rs_01",
            encryption="he",
            shape=(1,),
            dtype="int",
            data=[0],  # 随便填个data
            process="ss2he",
            key=pk_buffer,
            n_batches=2,
            uid=f"round{self.round}_1",
            router_table=self.router_table,
        )
        self.rc.send(pubkey_pack)
        logging.warning(
            f"\033[33m ROUND {self.round} pk_buffer {self.public_key} \033[0m"
        )

        packs = self.rc.recv(timeout=180)
        logging.warning(
            f"\033[33m ROUND {self.round} recv packs {[pack.uid for pack in packs]} \033[0m"
        )
        pack = packs[0]
        result = [
            self.kit.decryptor().decrypt_raw(
                pickle.loads(
                    ct_buffer.encode("latin1"), filter_type=pickle.FilterType.BLACKLIST
                )
            )
            for ct_buffer in pack.data
        ]
        global_weight = [
            data / 2**40 for data in result
        ]  # right shift 40 bits to recovery from big int
        logging.warning(
            f"\033[33m ROUND {self.round} recv global_weight:{global_weight} \033[0m"
        )

        self.round += 1

        return global_weight

    def update_weight(self, weights):
        # Input: [10*1_matrix(w0~w9), 10*1_matrix(w10~w19), 10*1_matrix(w20~w29)]

        # each party's weights are n*1 matrix
        # jax.tree_util.tree_flatten cannot flatten 2d-tensor to 1d-list
        weights_flatten = [item[0] for w in weights for item in w]

        # now weights_flatten is a flatten list of length 30: [w0~w29]
        weights_from_rs = self.sync_with_rs(weights_flatten)

        def unflatten():
            idx = 0
            for w in weights:
                yield np.array(weights_from_rs[idx : idx + len(w)]).reshape(-1, 1)
                idx += len(w)

        return list(unflatten())


# 通过该Hook类，启动隐私路由，对接华控清交平台的横向LR
class RouterLrAggrHook:
    def __init__(self, device):
        self.arbiter = device
        self.wa = WeightArbiter(device=device)

    def on_aggregate(self, weights: List[PYUObject]):
        """
        Hook on LR weights aggregation
        Args:
            weights: a list of PYUObject, each PYUObject points to an n*1 matrix, representing n weight values.
                For example, assuming that LR has 30 features and is scattered among 3 participants,
                the input list is: [PYUObject@Alice(w0~w9), PYUObject@Bob(w10~w19), PYUObject@Carol(w20~w29)]

        Returns: The new weights modified by hook in same format of input
        """

        # Send weights to arbiter
        new_weights = self.wa.update_weight(
            [w.to(self.arbiter) for w in weights], _num_returns=len(weights)
        )

        # Move weights to each party
        return [w.to(dev.device) for w, dev in zip(new_weights, weights)]
