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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sf_dataset import DualDomainSeqDataset, collate_fn_enhance
from sf_model import BERT4Rec, BAHE, BERT4RecEncoder, BERT4RecFusion
from sklearn.metrics import roc_auc_score
import argparse
import secretflow as sf
from secretflow import PYUObject, proxy
import logging
from pathlib import Path


@proxy(PYUObject)
class Client:
    def __init__(
        self, bahe_model, bert4rec_model, config, client_id, num_clients, device
    ):
        self.device = device
        self.bahe_model = device(bahe_model)
        self.bert4rec_model = device(bert4rec_model)
        self.client_id = device(client_id)
        self.config = config

    def _train_single_batch(self, batch):
        # 处理本地数据
        behavior_texts = batch['behavior_texts']
        seq = batch['seq_d1'] if self.client_id == 0 else batch['seq_d2']
        domain_id = batch['domain_id']

        # 使用BAHE生成用户嵌入
        user_embedding = self.bahe_model(behavior_texts)

        # 直接使用forward方法而不是process_sequence
        seq_embedding = self.bert4rec_model(
            batch['i_node'], seq, domain_id, user_embedding
        )

        return {'user_embedding': user_embedding, 'seq_embedding': seq_embedding}

    def update_gradients(self, loss):
        loss = self.device(loss)
        self.bahe_model.zero_grad()
        self.bert4rec_model.zero_grad()
        loss.backward()
        return True


@proxy(PYUObject)
class Server:
    def __init__(self, model, config, device):
        self.device = device
        self._model = device(model)
        self.config = config

    def _train_single_batch(self, client_outputs, batch):
        # 获取客户端输出
        user_embedding_d1 = client_outputs[0]['user_embedding']
        user_embedding_d2 = client_outputs[1]['user_embedding']
        seq_embedding_d1 = client_outputs[0]['seq_embedding']
        seq_embedding_d2 = client_outputs[1]['seq_embedding']

        # 使用forward方法而不是fusion_layer
        output = self._model(
            user_embedding_d1,
            user_embedding_d2,
            seq_embedding_d1,
            seq_embedding_d2,
            batch['domain_id'],
        )

        # 计算损失
        loss = nn.BCEWithLogitsLoss()(output, batch['label'])
        return loss


def sf_train(clients, server, epochs, train_dataset_params, batch_size):
    # 为alice和bob创建数据加载器
    alice_data = DualDomainSeqDataset(
        seq_len=train_dataset_params['seq_len'],
        isTrain=True,
        neg_nums=train_dataset_params['neg_nums'],
        long_length=train_dataset_params['long_length'],
        pad_id=train_dataset_params['pad_id'],
        csv_path=f"{train_dataset_params['dataset_type']}_dataset/{train_dataset_params['domain_type']}_train{int(train_dataset_params['overlap_ratio']*100)}.csv",
        domain_id=1,  # Alice负责域2
    )

    bob_data = DualDomainSeqDataset(
        seq_len=train_dataset_params['seq_len'],
        isTrain=True,
        neg_nums=train_dataset_params['neg_nums'],
        long_length=train_dataset_params['long_length'],
        pad_id=train_dataset_params['pad_id'],
        csv_path=f"{train_dataset_params['dataset_type']}_dataset/{train_dataset_params['domain_type']}_train{int(train_dataset_params['overlap_ratio']*100)}.csv",
        domain_id=0,  # Bob负责域1
    )

    alice_loader = DataLoader(
        alice_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_enhance,
    )

    bob_loader = DataLoader(
        bob_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_enhance,
    )

    for epoch in range(epochs):
        total_loss = 0

        for batch_id, (alice_batch, bob_batch) in enumerate(
            zip(alice_loader, bob_loader)
        ):
            # 获取客户端outputs
            client_outputs = []
            for client_id, (client, batch) in enumerate(
                zip(clients, [bob_batch, alice_batch])
            ):
                output = client._train_single_batch(batch)
                client_outputs.append(output.to(server.device))

            # 等待客户端计算完成
            sf.wait(client_outputs)

            # 准备服务器端需要的数据
            labels = {
                'label': server.device(bob_batch['label'].float()),
                'domain_id': server.device(bob_batch['domain_id']),
            }

            # 服务器端计算损失
            loss_pyu = server._train_single_batch(client_outputs, labels)
            loss = sf.reveal(loss_pyu)

            total_loss += loss.item()

            if batch_id % 10 == 0:
                logging.warning(
                    f'[Training Epoch: {epoch}] Batch: {batch_id}, Loss: {loss}'
                )

            # 更新客户端梯度
            updates = []
            for client in clients:
                ret = client.update_gradients(loss)  # 注意这里使用 loss_pyu
                updates.append(ret)

            sf.wait(updates)

        logging.warning(f"Training Epoch: {epoch}, total loss: {total_loss}")


if __name__ == '__main__':
    sf.init(["alice", "bob", "server"], address='local', debug_mode=False)
    alice_pyu = sf.PYU("alice")
    bob_pyu = sf.PYU("bob")
    server_pyu = sf.PYU("server")

    parser = argparse.ArgumentParser(description="BERT4Rec Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument(
        "--model_dir", type=str, default="model", help="Directory to save models"
    )
    parser.add_argument(
        "--log_file", type=str, default="train.log", help="Log file name"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="amazon_dataset", help="Path to dataset"
    )
    parser.add_argument('-ds', '--dataset_type', type=str, default='amazon')
    parser.add_argument('-dm', '--domain_type', type=str, default='cloth_sport')

    parser.add_argument(
        '--long_length',
        type=int,
        default=7,
        help='the length for setting long-tail node',
    )
    parser.add_argument(
        '--neg_nums', type=int, default=199, help='sample negative numbers'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=0.25,
        help='overlap ratio for choose dataset ',
    )
    # overlap_ratio = 0.25 表示25%的用户是跨域用户（即在两个域都有行为的用户）
    parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--bs', type=int, default=256, help='# images in batch')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden layer dim')
    parser.add_argument('--isInC', type=bool, default=False, help='add inc ')
    parser.add_argument('--isItC', type=bool, default=False, help='add itc')
    parser.add_argument('--ts1', type=float, default=0.5, help='mask rate for encoder')
    parser.add_argument('--ts2', type=float, default=0.5, help='mask rate for decoder')
    args = parser.parse_args()
    user_length = 895510  # 63275#6814 cdr23 #63275 cdr12
    item_length_d1 = 8240
    item_length_d2 = 26272
    item_length = 447410  # item_length_d1 + item_length_d2 + 1 + 20000#1739+2 #13713 cdr23 #1739 + 2#item_length_d1 + item_length_d2 + 1 + 20000#1739 + 1 +200 # 1 = pad item #item_length_d1 + item_length_d2 + 1 + 20000

    # 创建客户端模型
    bahe_model = BAHE(
        albert_model_name='albert/albert-base-v2',
        embed_dim=args.emb_dim,
        num_heads=4,
        ff_dim=512,
        num_layers=2,
    )

    bert4rec_encoder = BERT4RecEncoder(  # 需要新建一个BERT4RecEncoder类
        user_length=user_length,
        user_emb_dim=args.emb_dim,
        item_length=item_length,
        item_emb_dim=args.emb_dim,
        seq_len=args.seq_len,
        hid_dim=args.hid_dim,
    )

    # 创建服务器模型
    fusion_model = BERT4RecFusion(  # 需要新建一个BERT4RecFusion类
        embed_dim=args.emb_dim, num_heads=4, ff_dim=512
    )

    # 创建客户端和服务器
    clients = [
        Client(bahe_model, bert4rec_encoder, args, 0, 2, device=bob_pyu),  # Bob处理域1
        Client(
            bahe_model, bert4rec_encoder, args, 1, 2, device=alice_pyu
        ),  # Alice处理域2
    ]

    server = Server(fusion_model, args, device=server_pyu)

    # 训练参数
    train_dataset_params = {
        'seq_len': args.seq_len,
        'neg_nums': args.neg_nums,
        'long_length': args.long_length,
        'pad_id': item_length - 1,
        'dataset_type': args.dataset_type,
        'domain_type': args.domain_type,
        'overlap_ratio': args.overlap_ratio,
    }

    # 开始训练
    sf_train(
        clients,
        server,
        epochs=50,
        train_dataset_params=train_dataset_params,
        batch_size=32,
    )
