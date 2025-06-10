# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import secretflow as sf

from torchvision import datasets, transforms
from tqdm import tqdm

from darts.model import NetworkEMNIST
from dataSplit.mnist.mnist import split_mnist
from model.Server import Server
from model.Client import Client

import random
import torch
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.device_count())

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def add_args(parser):
    parser.add_argument(
        "--stage", type=str, default="search", help="stage: search; train"
    )
    parser.add_argument("--device", type=str, default="cpu", help="gpu")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        metavar="N",
        help="dataset used for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--client_number",
        type=int,
        default=20,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "--comm_round",
        type=int,
        default=50,
        help="how many round of communications we shoud use",
    )
    parser.add_argument("--layers", type=int, default=3, help="DARTS layers")
    parser.add_argument(
        "--dirichlet",
        type=float,
        default=0.6,
        help="狄利克雷分布的参数，用于分割数据集",
    )
    parser.add_argument("--num_classes", type=int, default=10, help="数据集类别个数")
    parser.add_argument("--temperature", type=float, default=5, help="设置蒸馏温度")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--local_points",
        type=int,
        default=5000,
        metavar="LP",
        help="the approximate fixed number of data points we will have on each local worker",
    )

    parser.add_argument(
        "--init_channels", type=int, default=16, help="num of init channels"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.025, help="init learning rate"
    )
    parser.add_argument(
        "--learning_rate_min", type=float, default=0.001, help="min learning rate"
    )

    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")

    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=3e-4,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )

    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument(
        "--lambda_train_regularizer",
        type=float,
        default=1,
        help="train regularizer parameter",
    )
    parser.add_argument(
        "--lambda_valid_regularizer",
        type=float,
        default=1,
        help="validation regularizer parameter",
    )
    parser.add_argument(
        "--report_freq", type=float, default=10, help="report frequency"
    )

    parser.add_argument("--tau_max", type=float, default=10, help="initial tau")
    parser.add_argument("--tau_min", type=float, default=1, help="minimum tau")

    parser.add_argument(
        "--auxiliary", action="store_true", default=False, help="use auxiliary tower"
    )
    parser.add_argument(
        "--arch", type=str, default="FedNAS_V1", help="which architecture to use"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    print(
        str(args.num_classes)
        + " "
        + str(args.client_number)
        + " "
        + str(args.comm_round)
    )

    # 训练设备
    device = torch.device(args.device)

    # 加载mnist训练集
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.MNIST(
        "./dataset", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./dataset", train=False, download=True, transform=transform
    )

    # 划分数据集给不同的客户端
    train_client_samples, test_client_samples = split_mnist(
        args.dirichlet, args.client_number
    )
    train_dataloader, test_dataloader = [], []
    search_dataloader, val_dataloader = [], []
    for i in range(args.client_number):
        random.shuffle(train_client_samples[i])
        half_size = len(train_client_samples[i]) // 2
        first_half = train_client_samples[i][:half_size]
        second_half = train_client_samples[i][half_size:]
        search_subset = torch.utils.data.Subset(train_dataset, indices=first_half)
        search_sub_loader = torch.utils.data.DataLoader(
            search_subset, batch_size=args.batch_size
        )
        val_subset = torch.utils.data.Subset(train_dataset, indices=second_half)
        val_sub_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size
        )
        train_subset = torch.utils.data.Subset(
            train_dataset, indices=train_client_samples[i]
        )
        train_sub_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size
        )
        test_subset = torch.utils.data.Subset(
            test_dataset, indices=test_client_samples[i]
        )
        test_sub_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=args.batch_size
        )
        search_dataloader.append(search_sub_loader)
        val_dataloader.append(val_sub_loader)
        train_dataloader.append(train_sub_loader)
        test_dataloader.append(test_sub_loader)

    # sf.shutdown()
    clients_name = ["client" + str(i + 1) for i in range(args.client_number)]
    print(clients_name)
    clients_id = []
    sf.init(clients_name, address="local", num_gpus=1, debug_mode=True)
    for i in clients_name:
        clients_id.append(sf.PYU(i))
    server_pyu = sf.PYU("server")

    server = Server(
        None, None, 60000, args.client_number, device, args, device=server_pyu
    )

    clients = []
    for client_id in range(args.client_number):
        clients.append(
            Client(
                client_id + 1,
                search_dataloader[client_id],
                val_dataloader[client_id],
                train_dataloader[client_id],
                test_dataloader[client_id],
                60000 / args.client_number,
                device,
                args,
                device=clients_id[client_id],
            )
        )

    global_model_params = None
    global_arch_params = None

    # 全局模型搜索args.comm_round轮
    print("**************开始搜索全局模型**************")
    for i in tqdm(range(args.comm_round)):
        weights_list, alphas_list, local_sample_num_list = [], [], []
        if global_model_params is None:
            global_model_params = server.get_model_weight()
            global_arch_params = server.get_arch_parameters()
        for j in range(len(clients)):
            clients[j].update_model(global_model_params.to(clients[j].device))
            clients[j].update_arch(global_arch_params.to(clients[j].device))
            start_time = time.time()
            clients[j].search()
            train_finished_time = time.time()
            print(
                "客户端索引：%d, 本地搜索时间: %d"
                % (j, train_finished_time - start_time)
            )
            client_weight = clients[j].get_weights()
            client_alphas = clients[j].get_alphas()
            client_local_sample_number = clients[j].get_local_sample_number()
            weights_list.append(client_weight.to(server.device))
            alphas_list.append(client_alphas.to(server.device))
            local_sample_num_list.append(client_local_sample_number.to(server.device))
        for j in range(len(clients)):
            server.add_local_trained_result(
                j, weights_list[j], alphas_list[j], local_sample_num_list[j]
            )
        server.aggregate()
    print("**************全局模型已经搜索完毕**************")
    print(
        "**************通过全局模型，指导客户端模型进行本地搜索与训练*****************"
    )
    for j in tqdm(range(len(clients))):
        clients[j].init_history_normal_reduce()
        server_weight = server.get_model_weight()
        clients[j].init_server_model(server_weight.to(clients[j].device))
        for m in range(10):
            start_time = time.time()
            clients[j].distillation_search()
            train_finished_time = time.time()
        genotype = clients[j].get_genotype().data
        search_model_weights = clients[j].get_weights().data
        model = NetworkEMNIST(
            args.init_channels, args.num_classes, args.layers, args.auxiliary, genotype
        )
        model.load_state_dict(search_model_weights, strict=False)
        clients[j].set_model(model)
        for n in range(500):
            clients[j].train()
            clients[j].test()
            test_acc = clients[j].get_test_acc().data
            test_loss = clients[j].get_test_loss().data
            with open("./result/mnist/client" + str(j) + ".txt", "a") as file:
                # 写入数据
                file.write(f"{test_acc} {test_loss}\n")
