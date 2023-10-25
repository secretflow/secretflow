# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import sys
import time

import multiprocess
import numpy as np
import torch

dispfl_root_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, dispfl_root_path)

from .model.vgg import vgg11
from .model.cnn_cifar10 import cnn_cifar10, cnn_cifar100
from .core.dispfl_api_runner import DisPFLAPI, SfDisPFLAPI
from .data.cifar10 import load_partition_data_cifar10
from .data.cifar100.data_loader import load_partition_data_cifar100
from .data.tiny_imagenet import load_partition_data_tiny
from .model.resnet import customized_resnet18, tiny_resnet18
from .core.my_model_trainer import MyModelTrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        metavar='N',
        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'",
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        metavar='N',
        help='dataset used for training',
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default=f'{dispfl_root_path} + data/',
        help='data directory, please feel free to change the directory to the right place',
    )

    parser.add_argument(
        '--partition_method',
        type=str,
        default='dir',
        metavar='N',
        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
        "one called 'n_cls' short for how many classes allocated for each client"
        "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution",
    )

    parser.add_argument(
        '--partition_alpha',
        type=float,
        default=0.3,
        metavar='PA',
        help='available parameters for data partition method',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        metavar='N',
        help='local batch size for training',
    )

    parser.add_argument(
        '--client_optimizer', type=str, default='sgd', help='SGD with momentum'
    )

    parser.add_argument(
        '--lr', type=float, default=0.1, metavar='LR', help='learning rate'
    )

    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.998,
        metavar='LR_decay',
        help='learning rate decay',
    )

    parser.add_argument('--wd', help='weight decay parameter', type=float, default=5e-4)

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        metavar='EP',
        help='local training epochs for each client',
    )

    parser.add_argument(
        '--client_num_in_total',
        type=int,
        default=100,
        metavar='NN',
        help='number of workers in a distributed cluster',
    )

    parser.add_argument(
        '--frac',
        type=float,
        default=0.1,
        metavar='NN',
        help='available communication fraction each round',
    )

    parser.add_argument(
        '--momentum', type=float, default=0, metavar='NN', help='momentum'
    )

    parser.add_argument(
        '--comm_round', type=int, default=10, help='total communication rounds'
    )

    parser.add_argument(
        '--frequency_of_the_test',
        type=int,
        default=1,
        help='the frequency of the test algorithms',
    )

    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    parser.add_argument('--ci', type=int, default=0, help='CI')

    parser.add_argument(
        '--dense_ratio', type=float, default=0.5, help='local density ratio'
    )

    parser.add_argument(
        '--anneal_factor', type=float, default=0.5, help='anneal factor for pruning'
    )

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--cs", type=str, default='v0')
    parser.add_argument("--active", type=float, default=1.0)

    parser.add_argument("--public_portion", type=float, default=0)
    parser.add_argument("--erk_power_scale", type=float, default=1)
    parser.add_argument("--dis_gradient_check", action='store_true')
    parser.add_argument("--strict_avg", action='store_true')
    parser.add_argument("--static", action='store_true')
    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--save_masks", action='store_true')
    parser.add_argument("--different_initial", action='store_true')
    parser.add_argument("--record_mask_diff", action='store_true')
    parser.add_argument("--diff_spa", action='store_true')
    parser.add_argument("--global_test", action='store_true')
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--run_type", type=str, default="none")
    return parser


def load_data(args, dataset_name, using_logger):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_cifar10(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            using_logger,
        )
    elif dataset_name == "cifar100":
        args.data_dir += "cifar100"
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_cifar100(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            using_logger,
        )
    elif dataset_name == "tiny":
        args.data_dir += "tiny_imagenet"
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_tiny(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            using_logger,
        )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def create_model(args, model_name, class_num):
    model = None
    if model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != 'tiny':
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    return model


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)


def logger_config(logger_path, logging_name):
    logging.basicConfig(
        filename=logger_path,
        filemode="w",
        encoding="UTF-8",
        format='[%(asctime)s]:%(message)s',
        level=logging.DEBUG,
        stream=None,
    )
    ret_logger = logging.getLogger(logging_name)
    ret_logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]:%(message)s')
    file_handler = logging.FileHandler(logger_path, mode='w', encoding='UTF-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    # terminal_handler = logging.StreamHandler()
    # terminal_handler.setLevel(level=logging.DEBUG)
    # terminal_handler.setFormatter(formatter)

    ret_logger.addHandler(file_handler)
    # ret_logger.addHandler(terminal_handler)
    return ret_logger


def run():
    parser = add_args(argparse.ArgumentParser(description='dispfl-standalone'))
    args = parser.parse_args()

    print("torch version{}".format(torch.__version__))
    if torch.cuda.is_available():
        print("cuda is avaliable, use cuda:" + str(args.gpu))
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    data_partition = args.partition_method
    if data_partition != "homo":
        data_partition += str(args.partition_alpha)
    args.identity = "dispfl" + "-" + args.dataset + "-" + data_partition
    args.identity += "-mdl" + args.model
    args.identity += "-cs" + args.cs

    if args.save_masks:
        args.identity += "-masks"

    if args.diff_spa:
        args.identity += "-diff_spa"

    if args.uniform:
        args.identity += "-uniform_init"
    else:
        args.identity += "-ERK_init"

    if args.different_initial:
        args.identity += "-diff_init"
    else:
        args.identity += "-same_init"

    if args.global_test:
        args.identity += "-g"

    if args.static:
        args.identity += '-RSM'
    else:
        args.identity += '-DST'

    args.identity += (
        "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    )
    args.client_num_per_round = int(args.client_num_in_total * args.frac)
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += "-dr" + str(args.dense_ratio)
    args.identity += "-active" + str(args.active)
    args.identity += '-seed' + str(args.seed)
    args.identity += '-run_type' + str(args.run_type)

    log_dir = os.path.abspath(os.getcwd()) + "/log/dispfl/" + args.dataset + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # log_path = log_dir + args.identity + '.log'
    log_path = log_dir + f"{args.dataset}-{args.run_type}-{args.comm_round}.log"
    # logger = logger_config(logger_path=log_path, logging_name=args.identity)
    logger = logger_config(logger_path=log_path, logging_name="myLogger")

    logger.info(args)
    logger.info(device)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.run_type and args.run_type == "sf":
        import secretflow as sf

        print("running in sf type.")
        # create secretflow device alice and bob
        sf.shutdown()
        print(f"client num is {args.client_num_in_total}")
        parties = [str(i) for i in range(args.client_num_in_total)]
        sf.init(
            parties=parties,
            address="local",
            num_cpus=multiprocess.cpu_count(),
            log_to_driver=True,
            omp_num_threads=multiprocess.cpu_count(),
            logging_level="debug",
        )
        sf_devices = [sf.PYU(party_name) for party_name in parties]

        # load data
        dataset = load_data(args, args.dataset, logger)

        # create model.
        model = create_model(args, model_name=args.model, class_num=len(dataset[-1][0]))
        model_trainer = custom_model_trainer(args, model, logger)
        dispflAPI = SfDisPFLAPI(
            dataset, device, args, model_trainer, logger, sf_devices
        )
        start_train_time = time.time()
        dispflAPI.train()
        train_time_used = time.time() - start_train_time

    else:
        print("runing in origin type.")
        # load data
        dataset = load_data(args, args.dataset, logger)
        # create model.
        model = create_model(args, model_name=args.model, class_num=len(dataset[-1][0]))
        model_trainer = custom_model_trainer(args, model, logger)
        dispflAPI = DisPFLAPI(dataset, device, args, model_trainer, logger, None)
        start_train_time = time.time()
        dispflAPI.train()
        train_time_used = time.time() - start_train_time
    print(f"total train use time = {train_time_used}s")


if __name__ == "__main__":
    run()
