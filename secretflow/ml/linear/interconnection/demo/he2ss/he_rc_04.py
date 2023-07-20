from anyconn_core import AppConfig
from router import RC, Pack
import pickle
from heu import phe
import logging

logger = logging.getLogger("anyconn")

router_table = {
    "rs_01": {"rs": ["rs_02"], "rc": ["rc_01", "rc_03"]},
    "rs_02": {"rs": ["rs_01"], "rc": ["rc_02", "rc_04"]},
}

nodes = [
    {"id": "rs_01", "tag": "RS", "address": "localhost:50051"},
    {"id": "rs_02", "tag": "RS", "address": "localhost:50052"},
    {"id": "rc_01", "tag": "RC01", "address": "localhost:50061"},
    {"id": "rc_02", "tag": "RC02", "address": "localhost:50062"},
    {"id": "rc_03", "tag": "RC", "address": "localhost:50063"},
    {"id": "rc_04", "tag": "RC", "address": "localhost:50064"},
]


def run_rc_04(x_list, server_he):
    rc = RC()
    with rc.run_in_thread(AppConfig(node_id="rc_04", nodes=nodes)):
        x2_list = [4, 5, 6]
        x_e_list = [server_he.encryptor().encrypt_raw(x) for x in x_list]
        x2_e_list = [server_he.encryptor().encrypt_raw(x) for x in x2_list]
        x1_e_list = [server_he.evaluator().sub(x_e, x2_e) for x_e, x2_e in zip(x_e_list, x2_e_list)]
        x1_e_enc_list = [pickle.dumps(x1_e) for x1_e in x1_e_list]
        logger.info("x1_e_list")

        pack1 = Pack(
            task_id="1",
            source_id="rc_04",
            data_id="2",
            target_id="rc_03",
            encryption="he",
            shape=(len(x1_e_enc_list),),
            dtype="int",
            data=x1_e_enc_list,
            process="he2ss",
            key=None,
            n_batches=1,
            uid=1,
            router_table=router_table,
        )
        logger.info("pack1!")

        rc.send(pack1)
        logger.info("pack1 sends successfully!")

        pack2 = Pack(
            task_id="1",
            source_id="rc_04",
            data_id="2",
            target_id="rs_02",
            encryption="ss",
            shape=(len(x2_list),),
            dtype="int",
            data=x2_list,
            process="he2ss",
            key=None,
            n_batches=1,
            uid=1,
            router_table=router_table,
        )
        rc.send(pack2)


if __name__ == "__main__":
    with open("public_key", "rb") as f:
        public_key = pickle.load(f)
    x_list = [1, 2, 3]
    server_he = phe.setup(public_key)
    run_rc_04(x_list, server_he)
