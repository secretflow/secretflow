import os
import torch
import torch.nn as nn
import yaml
import sys
import argparse
from torch.utils.data import DataLoader
sys.path.append("../")
from src.flbase.utils import setup_clients, resume_training
from src.utils import setup_seed, mkdirs, get_datasets, load_from_pkl, save_to_pkl
from src.flbase.strategies.FedAvg import FedAvgClient, FedAvgServer
from src.flbase.strategies.FedPCG import FedPCGClient,FedPCGServer
from src.flbase.strategies.FedNH import FedNHClient, FedNHServer
import secretflow as sf
from secretflow import PYUObject, proxy

global wandb_installed
try:
    import wandb
    wandb_installed = True
except ModuleNotFoundError:
    wandb_installed = False


def run(args):

    use_wandb = wandb_installed and args.use_wandb
    setup_seed(args.global_seed)

    with open(args.yamlfile, "r",encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    # parse the default setting
    server_config = config['server_config']
    client_config = config['client_config']

    # overwrite with inputs
    server_config['strategy'] = args.strategy
    server_config['num_clients'] = args.num_clients
    server_config['num_rounds'] = args.num_rounds
    server_config['participate_ratio'] = args.participate_ratio
    server_config['partition'] = args.partition
    server_config['beta'] = args.beta
    server_config['num_classes_per_client'] = args.num_classes_per_client
    server_config['num_shards_per_client'] = args.num_shards_per_client
    client_config['num_rounds'] = args.num_rounds
    client_config['global_seed'] = args.global_seed
    client_config['optimizer'] = args.optimizer
    client_config['client_lr'] = args.client_lr
    client_config['client_lr_scheduler'] = args.client_lr_scheduler
    client_config['sgd_momentum'] = args.sgd_momentum
    client_config['sgd_weight_decay'] = args.sgd_weight_decay
    client_config['use_sam'] = args.use_sam
    client_config['no_norm'] = args.no_norm

    if server_config['partition'] == 'noniid-label-distribution':
        partition_arg = f'beta:{args.beta}'
    elif server_config['partition'] == 'noniid-label-quantity':
        partition_arg = f'num_classes_per_client:{args.num_classes_per_client}'
    elif server_config['partition'] == 'shards':
        partition_arg = f'num_shards_per_client:{args.num_shards_per_client}'
    else:
        raise ValueError('not implemented partition')

    if args.strategy == 'FedAvg':
        ClientCstr, ServerCstr = FedAvgClient, FedAvgServer
        hyper_params = None

    elif args.strategy == 'FedNH':
        ClientCstr, ServerCstr = FedNHClient, FedNHServer
        server_config['FedNH_smoothing'] = args.FedNH_smoothing
        server_config['FedNH_server_adv_prototype_agg'] = args.FedNH_server_adv_prototype_agg
        client_config['FedNH_client_adv_prototype_agg'] = args.FedNH_client_adv_prototype_agg
        hyper_params = f"FedNH_smoothing:{args.FedNH_smoothing}_FedNH_client_adv_prototype_agg:{args.FedNH_client_adv_prototype_agg}"

    elif args.strategy == 'FedPCG':
        ClientCstr, ServerCstr = FedPCGClient, FedPCGServer
        server_config['FedNH_smoothing'] = args.FedNH_smoothing
        server_config['FedNH_server_adv_prototype_agg'] = args.FedNH_server_adv_prototype_agg
        client_config['FedNH_client_adv_prototype_agg'] = args.FedNH_client_adv_prototype_agg
        hyper_params = f"FedNH_smoothing:{args.FedNH_smoothing}_FedNH_client_adv_prototype_agg:{args.FedNH_client_adv_prototype_agg}"
    else:
        raise ValueError("Invalid strategy!")

    run_tag = f"{server_config['strategy']}_{server_config['dataset']}_{client_config['model']}_{server_config['partition']}_{partition_arg}_num_clients:{server_config['num_clients']}_participate_ratio:{server_config['participate_ratio']}_global_seed:{args.global_seed}"
    if hyper_params is not None:
        run_tag += "_" + hyper_params

    run_tag += f"_no_norm:{args.no_norm}"
    directory = f"./{args.purpose}_{server_config['strategy']}/"
    mkdirs(directory)
    # path = directory + run_tag
    path = directory
    if os.path.exists(path + '_final_server_obj.pkl'):
        print(f"Task:{run_tag} is finished. Exiting...")
        exit()
    print('results are saved in: ', path)
    # exit()
    if use_wandb:
        wandb.init(config=config, name=run_tag, project=args.purpose)

    client_config_lst = [client_config for i in range(args.num_clients)]
    criterion = nn.CrossEntropyLoss()

    trainset, testset, _ = get_datasets(server_config['dataset'])

    # setup clients
    if server_config['split_testset'] == False:
        clients_dict = setup_clients(ClientCstr, trainset, None, criterion,
                                     client_config_lst, args.device,
                                     server_config=server_config,
                                     beta=server_config['beta'],
                                     num_classes_per_client=server_config['num_classes_per_client'],
                                     num_shards_per_client=server_config['num_shards_per_client'],
                                     )
    else:
        print('split test set!')
        clients_dict = setup_clients(ClientCstr, trainset, testset, criterion,
                                     client_config_lst, args.device,
                                     server_config=server_config,
                                     beta=server_config['beta'],
                                     num_classes_per_client=server_config['num_classes_per_client'],
                                     num_shards_per_client=server_config['num_shards_per_client'],
                                     same_testset=False
                                     )
    # setup server and run
    if args.strategy != 'Local':
        server = ServerCstr(server_config, clients_dict, exclude=server_config['exclude'],
                            server_side_criterion=criterion, global_testset=testset, global_trainset=trainset,
                            client_cstr=ClientCstr, server_side_client_config=client_config,
                            server_side_client_device=args.device)
        print('Strategy Related Hyper-parameters:')
        print(' server side')
        for k in server_config.keys():
            if args.strategy in k:
                print(' ', k, ":", server_config[k])
        print(' client side')
        for k in client_config.keys():
            if args.strategy in k:
                print(' ', k, ":", client_config[k])
        server.run(filename=path + '_best_global_model.pkl', use_wandb=use_wandb, global_seed=args.global_seed)
        server.save(filename=path + '_final_server_obj.pkl', keep_clients_model=args.keep_clients_model)
    else:
        expected_num_rounds = int(server_config['num_rounds'] * server_config['participate_ratio'])
        init_weight = clients_dict[0].get_params()
        global_testloader = DataLoader(testset, batch_size=128, shuffle=False)
        for cid in clients_dict.keys():
            print(f"Progress:{cid}/{len(clients_dict)}")
            client = clients_dict[cid]
            client.set_params(init_weight, exclude_keys=set())
            for r in range(1, expected_num_rounds + 1):
                setup_seed(r + args.global_seed)
                client.training(r, client.client_config['num_epochs'])
                client.testing(r, global_testloader)
                print(f"Round: {r}/{expected_num_rounds}", client.test_acc_dict[r]['acc_by_criteria'])
            client.model = None
            client.trainloader = None
            client.trainset = None
            client.new_state_dict = None
        save_to_pkl(clients_dict, path + '_final_clients_dict.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Algorithms.')
    # general settings
    parser.add_argument('--purpose', default='experiments', type=str, help='purpose of this run')
    parser.add_argument('--device', default='cuda:1', type=str, help='cuda device')
    parser.add_argument('--global_seed', default=2022, type=int, help='Global random seed.')
    parser.add_argument('--use_wandb', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use wandb pkg')
    parser.add_argument('--keep_clients_model', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Keep FedAVG local model')
    # model architecture
    parser.add_argument('--no_norm', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use group/batch norm or not')
    # optimizer
    parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer')
    parser.add_argument('--num_epochs', default=5, type=int, help='num local epochs')
    parser.add_argument('--client_lr', default=0.1, type=float, help='client side initial learning rate')
    parser.add_argument('--client_lr_scheduler', default='stepwise', type=str, help='client side learning rate update strategy')
    parser.add_argument('--sgd_momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('--sgd_weight_decay', default=1e-5, type=float, help='sgd weight decay')
    parser.add_argument('--use_sam', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use SAM optimizer')
    # server config
    parser.add_argument('--yamlfile', default=None, type=str, help='Configuration file.')
    parser.add_argument('--strategy', default=None, type=str, help='strategy FL')
    parser.add_argument('--num_clients', default=100, type=int, help='number of clients')
    parser.add_argument('--num_rounds', default=200, type=int, help='number of communication rounds')
    parser.add_argument('--participate_ratio', default=0.1, type=float, help='participate ratio')
    parser.add_argument('--partition', default=None, type=str, help='method for partition the dataset')
    parser.add_argument('--beta', default=None, type=str, help='Dirichlet Distribution parameter')
    parser.add_argument('--num_classes_per_client', default=None, type=int, help='pathological non-iid parameter')
    parser.add_argument('--num_shards_per_client', default=None, type=int, help='pathological non-iid parameter fedavg simulation')

    # strategy parameters
    parser.add_argument('--FedNH_smoothing', default=0.9, type=float, help='moving average parameters')
    parser.add_argument('--FedNH_server_adv_prototype_agg', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH server adv agg')
    parser.add_argument('--FedNH_client_adv_prototype_agg', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH client adv agg')

    parser.add_argument('--FedROD_hyper_clf', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedRod phead uses hypernetwork')
    parser.add_argument('--FedROD_phead_separate', default=False, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help='FedROD phead separate train')
    parser.add_argument('--FedProto_lambda', default=0.1, type=float, help='FedProto local penalty lambda')
    parser.add_argument('--FedRep_head_epochs', default=10, type=int, help='FedRep local epochs to update head')
    parser.add_argument('--FedBABU_finetune_epoch', default=5, type=int, help='FedBABU local epochs to finetune')
    parser.add_argument('--Ditto_lambda', default=0.75, type=float, help='penalty parameter for Ditto')
    parser.add_argument('--CReFF_num_of_fl_feature', default=100, type=int, help='num of federated feature per class')
    parser.add_argument('--CReFF_match_epoch', default=100, type=int, help='epoch used to minmize gradient matching loss')
    parser.add_argument('--CReFF_crt_epoch', default=300, type=int, help='epoch used to retrain classifier')
    parser.add_argument('--CReFF_lr_net', default=0.01, type=float, help='lr for head')
    parser.add_argument('--CReFF_lr_feature', default=0.1, type=float, help='lr for feature')

    args = parser.parse_args()
    run(args)
