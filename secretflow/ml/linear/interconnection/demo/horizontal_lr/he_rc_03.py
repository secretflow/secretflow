from anyconn_core import AppConfig
from router import RC, Pack
from heu import phe
import logging
import pickle
from typing import List

import numpy as np

logger = logging.getLogger("anyconn")

router_table = {
    "rs_01": {"rs": ["rs_02"], "rc": ["rc_01", "rc_03"]},
    "rs_02": {"rs": ["rs_01"], "rc": ["rc_02", "rc_03"]},
}

nodes = [
    {"id": "rs_01", "tag": "RS", "address": "localhost:50051"},
    {"id": "rs_02", "tag": "RS", "address": "localhost:50052"},
    {"id": "rc_01", "tag": "RC01", "address": "localhost:50061"},
    {"id": "rc_02", "tag": "RC02", "address": "localhost:50062"},
    {"id": "rc_03", "tag": "RC03", "address": "localhost:50063"},
]

# nodes = [
#     {"id": "rs_01", "tag": "RS", "address": "172.16.112.233:31379"},
#     {"id": "rs_02", "tag": "RS", "address": "172.16.112.233:31918"},
#     {"id": "rc_01", "tag": "RC01", "address": "172.16.112.233:7777"},
#     {"id": "rc_02", "tag": "RC02", "address": "172.16.112.233:7777"},
#     {"id": "rc_03", "tag": "RC03", "address": "192.168.113.20:50063"},
# ]


class Arbiter:
    def __init__(self):
        self.kit = phe.setup(phe.SchemaType.ZPaillier, 2048)
        # You can get the public key by following method:
        self.public_key = self.kit.public_key()
        self.rc = RC()
        self.rc_id = "rc_03"
        self.rc.run_in_thread_forever(AppConfig(node_id=self.rc_id, nodes=nodes))
        self.router_table = router_table
        self.round = 0

    def sync_with_rs(self, flatten_weights: List[float]):
        # secretflow sends its own weights to secret-sharing platform
        w_int = [int(_w * 2 ** 40) for _w in flatten_weights]  # left shift 40 bits.
        w2_int = [int(_w * 2 ** 40) for _w in np.random.random(size=len(w_int))]
        w1_int = [_w - _w2 for _w, _w2 in zip(w_int, w2_int)]
        w1_pack = Pack(task_id='1',
                       source_id=self.rc_id,
                       data_id='2',
                       target_id="rs_01",
                       encryption='ss',
                       shape=(len(w1_int),),
                       dtype='int',
                       data=w1_int,
                       process="he2ss",
                       key=None,
                       n_batches=1,
                       uid=f"round{self.round}__2",
                       router_table=self.router_table)
        w2_pack = Pack(task_id='1',
                       source_id=self.rc_id,
                       data_id='2',
                       target_id="rs_02",
                       encryption='ss',
                       shape=(len(w2_int),),
                       dtype='int',
                       data=w2_int,
                       process="he2ss",
                       key=None,
                       n_batches=1,
                       uid=f"round{self.round}_2",
                       router_table=self.router_table)
        self.rc.send(w1_pack)
        logger.info(f"ROUND {self.round}  w1_int {w1_int}")
        self.rc.send(w2_pack)
        logger.info(f"ROUND {self.round}  w2_int {w2_int}")

        # secretflow receives aggregated global weights from secret-sharing platform.
        pk_buffer = pickle.dumps(self.public_key).decode('latin1')
        pubkey_pack = Pack(
            task_id='1',
            source_id=self.rc_id,
            data_id='2',
            target_id="rs_01",
            encryption='he',
            shape=(1,),
            dtype='int',
            data=[0],  # 随便填个data
            process="ss2he",
            key=pk_buffer,
            n_batches=2,
            uid=f"round{self.round}_1",
            router_table=self.router_table)
        self.rc.send(pubkey_pack)
        # logger.info(f"ROUND {self.round} pk_buffer {self.public_key}")

        packs = self.rc.recv()
        logger.info(f"ROUND {self.round} recv packs {[pack.uid for pack in packs]}")
        pack = packs[0]
        result = [self.kit.decryptor().decrypt_raw(pickle.loads(ct_buffer.encode("latin1")))
                  for ct_buffer in pack.data]
        agg_weight = [data / 2 ** 40 for data in result]  # right shift 40 bits to recovery from big int
        logger.info(f"ROUND {self.round} recv agg_weight:{agg_weight}")
        self.round += 1
        return agg_weight


if __name__ == "__main__":
    arbiter = Arbiter()
    w = [i for i in np.arange(31) + 0.5]
    for i in range(4):
        w = arbiter.sync_with_rs(w)
        logger.info(f"round {i} global_w: {w}")
