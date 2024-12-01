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

import logging
import secretflow as sf


def run_fl(clients, server, args):
    loading = []
    for c in clients:
        loading.append(c.load_dataset())
    ################################################################
    # 隐语中都是异步执行的，这里的 sf.wait 表示等待所有 client 加载数据完成
    ################################################################
    sf.wait(loading)
    # Initialize the aggretation weight
    if args.do_eval:
        for client in clients:
            client.evaluation(mode="test")
    else:
        for round in range(1, args.num_round + 1):
            client_models = []

            # Train with these clients
            for client in clients:
                # Train one client
                client.train_epoch(round, args)
                weights = client.get_reps_shared()
                client_models.append(weights.to(server.device))

            global_model = server.aggregate_reps(client_models)

            setting = []
            for client in clients:
                ret = client.set_global_reps(global_model.to(client.device))
                setting.append(ret)
            ########################
            # 等待一轮训练完成
            ########################
            sf.wait(setting)
            if round % args.eval_interval == 0:
                for client in clients:
                    logging.warning("Epoch%d Valid:" % round)
                    client.evaluation(mode="valid")
        for client in clients:
            client.evaluation(mode="test")
