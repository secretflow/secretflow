#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import secretflow as sf
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import Client, Server
from models import CNNMnist, CNNFemnist, CNNCifar, Lenet
from utils import get_dataset, exp_details

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def calculate_avg(total,length):
    return total / length
def add_and_div(loss,length):
    return sum(loss) / length
def mean(list_a):
    return np.mean(list_a)
def std(list_a):
    return np.std(list_a)

def FedProto_Secretflow(args, train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, clients, server,
                       server_pyu):
    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    local_model_list = []
    for round in tqdm(range(args.rounds)):
        local_weights, local_weights_global, local_losses, local_protos = [], [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')
        for idx, local_model in enumerate(clients):
            local_model.update_weights_het(args, idx, global_round=round)
            w = local_model.get_param_model()
            loss = local_model.get_loss()
            acc = local_model.get_acc()
            protos = local_model.get_protos()
            agg_protos = local_model.agg_func(protos).to(server.device)
            #             agg_protos = {1:[512,1,1],7:[512,1,1],0:[512,1,1],9:[512,1,1],}
            local_weights.append(copy.deepcopy(w))
            local_weights_global.append(copy.deepcopy(w.to(server.device)))
            local_losses.append(copy.deepcopy(local_model.get_local_loss()).to(server.device))
            local_protos[idx] = agg_protos

        local_weights_list = local_weights
        local_weights_list_global = local_weights_global

        for idx, local_model in enumerate(clients):
            local_model.set_weights(local_weights_list[idx])
        #             local_model_list.append(local_model.to(server.device))
        # update global weights dict:{10}={1:[512,1,1],...}
        #         setting = []
        global_protos = server.proto_aggregation(local_protos)
        for local_model in clients:
            local_protos = global_protos.to(local_model.device)
            local_model.set_global_protos(local_protos)

        #             setting.append(ret)
        #         sf.wait(setting)
        loss_avg = server_pyu(add_and_div)(local_losses, len(local_losses))
        train_loss.append(loss_avg)
    global_protos = global_protos.to(server.device)
    #     local_weights_list=local_weights_list.to(server.device)
    server.test_inference_new_het_lt(args, local_weights_list_global, test_dataset, classes_list, user_groups_lt,
                                     global_protos)
    acc_list_l = server.get_acc_list_l()
    acc_list_g = server.get_acc_list_g()
    loss_list = server.get_loss_list()
    print('For all users (with protos), mean of test acc is ', sf.reveal(server_pyu(mean)(acc_list_g)),
          'std of test acc is ', sf.reveal(server_pyu(std)(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is ', sf.reveal(server_pyu(mean)(acc_list_l)),
          'std of test acc is ', sf.reveal(server_pyu(std)(acc_list_l)))
    print('For all users (with protos), mean of proto loss is ', sf.reveal(server_pyu(mean)(loss_list)),
          'std of test acc is ', sf.reveal(server_pyu(std)(loss_list)))


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # Build models
    sf.shutdown()
    sf.init(["client_1", "client_2", "client_3", "client_4", "client_5", "client_6", "client_7", "client_8", "client_9",
             "client_10", "client_11", "client_12", "client_13", "client_14", "client_15", "client_16", "client_17",
             "client_18", "client_19", "client_20", "server"], address='local', num_gpus=1)
    # sf.init(["client_1", "client_2", "client_3", "client_4", "client_5", "client_6", "client_7", "client_8", "server"], address='local', num_gpus=1)
    clients = []
    for i in np.arange(args.num_users):
        client_i = "client_" + str(i + 1)
        print(client_i)
        client_i_pyu = sf.PYU(client_i)
        client_i = Client(args=args, dataset=train_dataset, idxs=user_groups[i], device=client_i_pyu)

        clients.append(client_i)
    server_pyu = sf.PYU("server")
    server = Server(args, device=server_pyu)
    print("clients", clients)
    for idx, local_model in enumerate(clients):
        print(idx, local_model)

    FedProto_Secretflow(args, train_dataset, test_dataset, user_groups, user_groups_lt, classes_list,clients, server,server_pyu)