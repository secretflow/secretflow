from anyconn_core import AppConfig
from router import RC, Pack
from heu import phe
import pickle
import logging

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


def run_rc_03():
    client_he = phe.setup(phe.SchemaType.ZPaillier, 2048)
    public_key = client_he.public_key()
    pk_buffer = pickle.dumps(public_key).decode('latin1')

    rc = RC()
    with rc.run_in_thread(AppConfig(node_id="rc_03", nodes=nodes)):
        arr = [1]
        new_pack = Pack(
            task_id='1',
            source_id="rc_03",
            data_id='2',
            target_id="rs_01",
            encryption="he",
            shape=(1,),
            dtype="int",
            data=arr,
            process="ss2he",
            key=pk_buffer,
            n_batches=2,
            uid="1",
            router_table=router_table,
        )
        rc.send(new_pack)
        logger.info("rc_03 send public_key pack.")

        result = []
        for pack in rc.recv():
            res = [client_he.decryptor().decrypt_raw(pickle.loads(ct_buffer.encode("latin1"))) for ct_buffer in
                   pack.data]
            result.extend(res)
        result = [data / 2 ** 40 for data in result]
        logger.info(f"rc_03 recv: {result}")


if __name__ == "__main__":
    run_rc_03()
