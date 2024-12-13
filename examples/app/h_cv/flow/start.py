"""
启动文件：python start.py --client_num 20 --device cuda --batch_size 1024 --epoch 4000 --inner_epoch 3 --dirichlet 0.6
"""
import argparse
import secretflow as sf

from client.cifar10 import Client
from dataSplit.cifar10 import split_cifar10
from server.FedAvg import Server, train
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())



parser = argparse.ArgumentParser()
parser.add_argument('--client_num', type=int, default=20)
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--inner_epoch', type=int, default=3)
parser.add_argument('--dirichlet', type=float, default=0.6)
args = parser.parse_args()

sf.shutdown()
clients_name = ['client' + str(i+1) for i in range(args.client_num)]
print(clients_name)
clients_id = []
sf.init(clients_name, address='local', num_gpus=1, debug_mode=True)
for i in clients_name:
    clients_id.append(sf.PYU(i))
server_pyu = sf.PYU("server")

train_client_samples, test_client_samples = split_cifar10(args.dirichlet, args.client_num)
clients = [Client(i+1, args.device, train_client_samples[i], test_client_samples[i], args.batch_size, args.inner_epoch, device=clients_id[i]) for i in range(args.client_num)]
# clients = [Client(i+1, args.device, train_client_samples[i], test_client_samples[i], args.batch_size, args.inner_epoch) for i in range(args.client_num)]
print(clients)
server = Server(device=server_pyu, epoch=args.epoch)
# server = Server(epoch=args.epoch)
train(clients, server, epoch=args.epoch)
