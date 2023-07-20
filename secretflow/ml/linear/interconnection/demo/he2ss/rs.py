from anyconn_core import AppConfig
from router import RS
from multiprocessing import Pool
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


def run_rs(_id: str):
    rs = RS()
    rs.run_node(AppConfig(node_id=_id, nodes=nodes))


def main():
    with Pool(2) as pool:
        pool.map(run_rs, ["rs_01", "rs_02"])
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
