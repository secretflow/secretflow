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
import secretflow as sf
import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import logging

from sl_dcn_torch import DeepCrossbase, DeepCrossfuse
from utils import getTrainData
from client import Client
from server import Server


def sf_train(clients, server, epochs, train_dataset, batch_size):

    ################################################################
    # 隐语中都是异步执行的，这里的 sf.wait 表示等待
    ################################################################

    for epoch in range(epochs):

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        total = 0
        for batch_id, (alice_data, bob_data, labels) in enumerate(train_loader):
            client_models = []
            for client in clients:

                client_id_pyu_obj = client.get_client_id()
                client_id = sf.reveal(client_id_pyu_obj)

                if client_id == 0:
                    weights = client._train_single_batch(alice_data)
                else:
                    weights = client._train_single_batch(bob_data)

                ################################################################
                # weights.to(server.device) 表示把 weights 从 client 发送到 server
                ################################################################
                client_models.append(weights.to(server.device))

            sf.wait(client_models)

            loss_pyu = server._train_single_batch(client_models, labels)
            loss, _ = sf.reveal(loss_pyu)

            total += loss.item()
            logging.warning(
                '[Training Epoch: {}] Batch: {}, Loss: {}'.format(epoch, batch_id, loss)
            )
            # global_weights = server.get_weights()  # 从服务器获取全局权重
            setting = []
            for client in clients:
                ret = client.update_gradients(
                    loss
                )  # Client 端根据 server 的 loss 更新梯度
                setting.append(ret)
            ########################
            # 等待一批训练完成
            ########################

            sf.wait(setting)
        logging.warning("Training Epoch: %d, total loss: %f" % (epoch, total))
        print("Epoch: %d is finished!" % (epoch))
        print("-------------------end-----------------------------")


if __name__ == '__main__':
    sf.shutdown()
    sf.init(["alice", "bob", "server"], address='local', num_gpus=0)
    alice_pyu = sf.PYU("alice")
    bob_pyu = sf.PYU("bob")
    server_pyu = sf.PYU("server")

    ################################################################
    # device=alice_pyu 表示这个 worker 是 alice 这一方的
    ################################################################

    deepcrossbase_config = {
        'deep_layers': [256, 32],  # 设置Deep模块的隐层大小
        'num_cross_layers': 4,  # cross模块的层数
        'batch_size': 32,
        'lr': 1e-3,
        'l2_regularization': 1e-4,
        'device_id': 0,
        'use_cuda': False,
        'train_file': 'data/train_set1.csv',
        'fea_file': 'data/max_fea_col.npy',
        'validate_file': 'data/val_set1.csv',
        'test_file': 'data/test_set1.csv',
        'model_name': 'deepcrossbase.model',
    }
    deepcrossfuse_config = {
        'deep_layers': [256, 256, 32],  # 设置Deep模块的隐层大小
        'dnn_input_dim': 64,  # Deep模块输入的大小
        'lr': 1e-3,
        'l2_regularization': 1e-4,
        'device_id': 0,
        'use_cuda': False,
        'train_file': 'data/train_set1.csv',
        'validate_file': 'data/val_set1.csv',
        'test_file': 'data/test_set1.csv',
        'model_name': 'deepcrossfuse.model',
    }
    columns_for_alice = [
        'I1',
        'I3',
        'I5',
        'I7',
        'I9',
        'I11',
        'I13',
        'C1',
        'C3',
        'C5',
        'C7',
        'C9',
        'C11',
        'C13',
        'C15',
        'C17',
        'C19',
        'C21',
        'C23',
        'C25',
    ]

    columns_for_bob = [
        'I2',
        'I4',
        'I6',
        'I8',
        'I10',
        'I12',
        'C2',
        'C4',
        'C6',
        'C8',
        'C10',
        'C12',
        'C14',
        'C16',
        'C18',
        'C20',
        'C22',
        'C24',
        'C26',
        'Label',
    ]

    client_num = 2
    # df_data_alice,labels, dense_features_cols_alice, sparse_features_cols_alice = getTrainData(deepcrossbase_config['train_file'], deepcrossbase_config['fea_file'],columns_for_alice)
    (
        df_data_alice,
        df_data_bob,
        labels,
        dense_features_cols_alice,
        sparse_features_cols_alice,
        dense_features_cols_bob,
        sparse_features_cols_bob,
    ) = getTrainData(
        deepcrossbase_config['train_file'],
        deepcrossbase_config['fea_file'],
        columns_for_alice,
        columns_for_bob,
    )

    client_model_alice = DeepCrossbase(
        deepcrossbase_config,
        dense_features_cols=dense_features_cols_alice,
        sparse_features_cols=sparse_features_cols_alice,
    )
    client_model_bob = DeepCrossbase(
        deepcrossbase_config,
        dense_features_cols=dense_features_cols_bob,
        sparse_features_cols=sparse_features_cols_bob,
    )
    server_model = DeepCrossfuse(
        deepcrossfuse_config,
        dense_features_cols_alice,
        dense_features_cols_bob,
        sparse_features_cols_alice,
        sparse_features_cols_bob,
    )
    train_dataset = Data.TensorDataset(
        torch.tensor(df_data_alice).float(),
        torch.tensor(df_data_bob).float(),
        torch.tensor(labels).float(),
    )

    clients = [
        Client(client_model_alice, deepcrossbase_config, 0, 2, device=alice_pyu),
        Client(client_model_bob, deepcrossbase_config, 1, 2, device=bob_pyu),
    ]

    server = Server(server_model, deepcrossfuse_config, device=server_pyu)

    sf_train(clients, server, 30, train_dataset, deepcrossbase_config['batch_size'])
