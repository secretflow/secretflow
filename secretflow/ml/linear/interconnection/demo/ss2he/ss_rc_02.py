from anyconn_core import AppConfig
import logging
from router import RC, Pack
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


def run_rc_02(data_slice: list):
    rc = RC()
    with rc.run_in_thread(AppConfig(node_id="rc_02", nodes=nodes)):
        new_pack = Pack(
            task_id='1',
            source_id="rc_02",
            data_id='2',
            target_id="rs_02",
            encryption="ss",
            shape=np.shape(data_slice),
            dtype="int",
            data=data_slice,
            process="ss2he",
            key=None,
            n_batches=2,
            uid="_1",
            router_table=router_table,
        )
        rc.send(new_pack)


if __name__ == "__main__":
    data_slice = [-1.5, -3.5, -5.5]
    data_slice = [int(data * 2 ** 40) for data in data_slice]
    run_rc_02(data_slice)
