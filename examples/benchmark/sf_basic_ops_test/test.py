# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import spu
from jax import numpy as jnp

import secretflow as sf
from secretflow.device import PYU, PYUObject, proxy
from secretflow.device.driver import wait
from secretflow.stats.united_stats import united_mean_and_var, united_median, united_var

# change this!!!
secretflow_config = {
    "alice": {
        # change ip and ports here
        "ip": "123.45.67.890",
        "port": 10011,
        "spu_port": 10010,
    },
    "bob": {
        # change ip and ports here
        "ip": "123.45.67.891",
        "port": 10044,
        "spu_port": 10010,
    },
}

# you may change this if you understand how this script works
trace_memory = False
data_set = None
g_data_len = 10**8
g_chunk_size = 2**16

protocol_dict = {
    "aby3": (spu.ProtocolKind.ABY3, 3),
    "cheetah": (spu.ProtocolKind.CHEETAH, 2),
    "semi2k": (spu.ProtocolKind.SEMI2K, 2),
}

data_index_config = {
    "alice": 1,
    "bob": 2,
    "carol": 3,
}


def make_chunk(lst, chunk_size):
    ret = []
    chunk_cnt = (len(lst) + chunk_size - 1) // chunk_size
    for i in range(chunk_cnt):
        end_index = min(len(lst), i * chunk_size + chunk_size)
        ret.append(lst[i * chunk_size : end_index])
    return ret


def get_sf_init_config(party, party_num):
    ret = {}
    ret["self_party"] = party
    ret["parties"] = {}
    for p, v in secretflow_config.items():
        ret["parties"][p] = {
            "address": f"{v['ip']}:{v['port']}",
            "listen_addr": f"0.0.0.0:{v['port']}",
        }
        if len(ret["parties"]) == party_num:
            break
    return ret


def get_spu_init_config(protocol: str, field: int):
    field_dic = {64: spu.FieldType.FM64, 128: spu.FieldType.FM128}
    nodes = []
    nodes_expect_len = protocol_dict[protocol][1]
    parties = list(secretflow_config.keys())
    for i in range(nodes_expect_len):
        nodes.append(
            {
                "party": parties[i],
                "id": f"local:{i}",
                "address": f"{secretflow_config[parties[i]]['ip']}:{secretflow_config[parties[i]]['spu_port']}",
            }
        )
    return {
        "nodes": nodes,
        "runtime_config": {
            "protocol": protocol_dict[protocol][0],
            "field": field_dic[field],
        },
    }


def init_sf(party: str, party_num: int):
    config = get_sf_init_config(party, party_num)
    sf.init(
        cluster_config=config,
        cross_silo_comm_backend="brpc_link",
        cross_silo_comm_options={
            "timeout_in_ms": 7200 * 1000,
            "recv_timeout_ms": 7200 * 1000,
            "http_timeout_ms": 7200 * 1000,
        },
        enable_waiting_for_other_parties_ready=True,
    )
    return sf


def get_spu(protocol: str, field: int):
    return sf.SPU(
        get_spu_init_config(protocol, field),
        link_desc={"recv_timeout_ms": 7200 * 1000, "http_timeout_ms": 7200 * 1000},
    )


def get_100_million_float():
    return jnp.array(np.random.uniform(1, 1000, 10**2))


from threading import Thread

import psutil


class MemoryTracer:
    def __init__(self):
        self.started = False

    def trace(self):
        res_max = 0
        init_use = psutil.virtual_memory().used
        with open(f"logs/memory_trace_{time.time()}.log", "w") as ofile:
            while self.started:
                cur_mem = psutil.virtual_memory()
                print(
                    f"MemoryTracer: TOTAL USED {cur_mem.used / (2**30):.3f}GB, PERCENT {cur_mem.percent}, TEST USED {(cur_mem.used - init_use) / (2**30):.3f}GB,",
                    file=ofile,
                )
                time.sleep(0.1)

    def start(self):
        if self.started:
            return
        self.started = True
        self.t = Thread(target=self.trace)
        self.t.start()

    def stop(self):
        if self.started:
            self.t.join(1)


class Clock:
    def __init__(self, label: str = "", expect_cnt=-1):
        self.start = time.time_ns()
        self.last = self.start
        self.expect_cnt = expect_cnt
        self.start_time = time.ctime()
        self.step_dict = defaultdict(lambda: [])
        print(f"{label} clock({self.start_time}): start...", flush=True)

    def point(self, key: str = ""):
        # last first
        now = time.time_ns()
        step = now - self.last
        start = now - self.start
        self.last = now
        self.step_dict[key].append(step)
        cur_step = len(self.step_dict[key])
        final_tag = ""
        if cur_step == self.expect_cnt:
            final_tag = "FINAL"
        step_str = "{:.2f}ms".format(step / 1e6)
        start_str = "{:.2f}ms".format(start / 1e6)
        step_sum_str = "{:.2f}ms".format(sum(self.step_dict[key]) / 1e6)
        print(
            f"clock({self.start_time}): {key}:{final_tag}  <step_sum: {step_sum_str}> <step: {step_str}> <since_start: {start_str}>"
        )


def make_chunk_index(length: int, chunk_size: int):
    return [(i, min(length, i + chunk_size)) for i in range(0, length, chunk_size)]


@proxy(PYUObject)
class DataLoader:
    def __init__(self, party: str):
        self.party = party
        data_set_dir = f"../data/basic{data_index_config[party]}"
        # assert os.path.exists(data_set_dir), f'{data_set_dir} not exixts'
        print("start reading...", flush=True)
        if os.path.exists(data_set_dir):
            self.data_set = pd.read_csv(
                data_set_dir,
                dtype={
                    f"X{data_index_config[party] * 2 - 1}": np.float32,
                    f"X{data_index_config[party] * 2}": np.float32,
                },
            )
        else:
            self.data_set = None

    def load_chunck(self, func_name, start, end):
        col_index = data_index_config[self.party] * 2 - 1
        if "var" in func_name or "median" in func_name:
            col_index += 1
        col_name = f"X{col_index}"
        if self.data_set is not None:
            return self.data_set.loc[start : end - 1, col_name].to_numpy()
        else:
            return np.random.randn(end - start).astype(np.float32)


def load_chuncks(party: str, func_name: str, length, chunk_size):
    loader = DataLoader(party, device=PYU(party))
    ret = []
    chunk_index = make_chunk_index(length, chunk_size)
    print(f"{party} get {len(chunk_index)} chunks, start ...", flush=True)
    for i, j in chunk_index:
        ret.append(loader.load_chunck(func_name, i, j))
        wait(ret[-1])
    print(f"{party} get {len(ret)} chunks, end ...", flush=True)
    return ret


def test_func_2(sf, spu, test_func, protocol):
    if protocol == "cheetah" and test_func == jnp.less:
        xs = load_chuncks("alice", str(test_func), g_data_len, g_chunk_size)
        ys = load_chuncks("bob", str(test_func), g_data_len, g_chunk_size)
    else:
        xs = load_chuncks("alice", str(test_func), g_data_len, g_data_len)
        ys = load_chuncks("bob", str(test_func), g_data_len, g_data_len)

    clock = Clock(f"test_func_2 {test_func}", len(xs))
    res = []
    for x, y in zip(xs, ys):
        if test_func in [jnp.add, jnp.multiply, jnp.less]:
            s_x = x.to(spu)
            s_y = y.to(spu)
            # clock.point("to(spu)")
            s_res = spu(lambda a, b: test_func(a, b))(s_x, s_y)
        else:
            if protocol == "cheetah" and test_func == var:
                s_res = test_func([x, y], spu, 50000)
            else:
                s_res = test_func([x, y], spu)
        wait(s_res)
        clock.point(f"spu {test_func}")
        res = sf.reveal(s_res)
        clock.point(f"reveal")

    return res


def test_func_3(sf, spu, test_func):
    xs = load_chuncks("alice", str(test_func), g_data_len, g_data_len)
    ys = load_chuncks("bob", str(test_func), g_data_len, g_data_len)
    zs = load_chuncks("carol", str(test_func), g_data_len, g_data_len)

    res = []
    clock = Clock(f"test_func_3 {test_func}", len(xs))
    for x, y, z in zip(xs, ys, zs):
        if test_func in [add_3, multiply_3]:
            s_x = x.to(spu)
            s_y = y.to(spu)
            s_z = z.to(spu)
            wait([s_x, s_y, s_z])
            clock.point("to(spu)")
            s_res = spu(test_func)(s_x, s_y, s_z)
        else:
            s_res = (
                test_func([x, y, z], spu, 50000)
                if test_func in [var]
                else test_func([x, y, z], spu)
            )
        wait(s_res)
        clock.point(f"spu {test_func}")
        res.append(sf.reveal(s_res))
        clock.point(f"reveal")

    return res


def var(arrs, compute_device, block_size=100000):
    return united_var(arrs, compute_device, block_size)


def var_reveal_mean(arrs, compute_device):
    _, var = united_mean_and_var(arrs, compute_device)
    return var


def median(arrs, compute_device):
    return united_median(arrs, compute_device)


def add_3(a, b, c):
    return jnp.add(a, jnp.add(b, c))


def multiply_3(a, b, c):
    return jnp.multiply(a, jnp.multiply(b, c))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="secretflow_basic_test.py", description="do basic test", epilog="enjoy"
    )
    parser.add_argument("--party")
    parser.add_argument("--protocol", default="aby3", type=str)
    parser.add_argument("--field", default=64, type=int)
    parser.add_argument("--func", default=".*", type=str)
    parser.add_argument("--trace_memory", default=False, type=bool)

    args = parser.parse_args()

    if args.trace_memory:
        trace_memory = True
        tracer = MemoryTracer()
        tracer.start()
        print("Use Trace Memory, this will cause preformance data discorrect !!!")
    try:
        assert args.protocol in list(
            protocol_dict.keys()
        ), f"{args.protocol} not in {list(protocol_dict.keys())}"
        print("Exe test with: ", args, flush=True)
        party_num = protocol_dict[args.protocol][1]
        sf = init_sf(args.party, party_num)
        spu = get_spu(args.protocol, args.field)

        if party_num == 2:
            funcs_2 = [jnp.add, jnp.multiply, jnp.less, var, var_reveal_mean, median]
            # funcs_2 = [jnp.add, var, median]
            print(f"All TestFuncs: {funcs_2}")
            funcs_2 = [func for func in funcs_2 if re.search(args.func, str(func))]
            print(f"TestFuncs: {funcs_2}")
            for func in funcs_2:
                test_func_2(sf, spu, func, args.protocol)
        else:
            funcs_3 = [add_3, multiply_3, var, var_reveal_mean, median]
            funcs_3 = [func for func in funcs_3 if re.search(args.func, str(func))]
            print(f"TestFuncs: {funcs_3}")
            for func in funcs_3:
                test_func_3(sf, spu, func)

        sf.shutdown()
    finally:
        if args.trace_memory:
            tracer.stop()
