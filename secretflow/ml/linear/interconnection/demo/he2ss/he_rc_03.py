from anyconn_core import AppConfig
from router import RC, Pack
from heu import phe
import pickle
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
    {"id": "rc_03", "tag": "RC", "address": "localhost:50063"},  # tag相同的会互相连接，因为rc3和rc4需要发送数据，所以需要设置为一样的tag
    {"id": "rc_04", "tag": "RC", "address": "localhost:50064"},
]


def run_rc_03():
    client_he = phe.setup(phe.SchemaType.ZPaillier, 2048)
    with open("public_key", 'wb') as f:
        pickle.dump(client_he.public_key(), f)

    rc = RC()
    with rc.run_in_thread(AppConfig(node_id="rc_03", nodes=nodes)):
        pack = rc.recv()[0]
        x1_list = [client_he.decryptor().decrypt_raw(pickle.loads(ct_x)) for ct_x in pack.data]
        logger.info(f"x1_list:{x1_list}")
        pack = Pack(
            task_id="1",
            source_id="rc_03",
            data_id="2",
            target_id="rs_01",
            encryption="ss",
            shape=(len(x1_list),),
            dtype="int",
            data=x1_list,
            process="he2ss",
            key=None,
            n_batches=1,
            uid="_1",
            router_table=router_table,
        )

        rc.send(pack)


if __name__ == "__main__":
    run_rc_03()
