import sys
from ExtractFeature import extract
sys.path.append('../')
from modelUtil import *
from FedUser import CDPUser, LDPUser, opacus
from FedServer import CDPServer, LDPServer
import numpy as np
import argparse
import time
import os
import pandas as pd


start_time = time.time()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10',
                        choices=['cifar10','cifar100'])
    parser.add_argument('--nclient', type=int, default= 100)
    parser.add_argument('--nclass', type=int, help= 'the number of class for this dataset', default= 10)
    parser.add_argument('--ncpc', type=int, help= 'the number of class assigned to each client', default=2)
    parser.add_argument('--encoder', type=str, default='clip', choices=['resnext', 'clip', 'simclr'])
    parser.add_argument('--model', type=str, default='linear_model_DN_IN', choices = ['linear_model_DN_IN'])
    parser.add_argument('--mode', type=str, default= 'LDP')
    parser.add_argument('--round',  type = int, default= 20)
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--sr',  type=float, default=1.0,
                        help='sample rate in each round')
    parser.add_argument('--lr',  type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--flr',  type=float, default=5e-2,
                        help='learning rate')
    parser.add_argument('--physical_bs', type = int, default=4, help= 'the max_physical_batch_size of Opacus LDP, decrease to 1 if cuda out of memory')
    parser.add_argument('--E',  type=int, default=2,
                        help='the index of experiment in AE')
    parser.add_argument('--bs',  type=int, default=64,
                        help='batch size')
    args = parser.parse_args()
    return args

args = parse_arguments()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_NAME = args.data
NUM_CLIENTS = args.nclient
NUM_CLASSES = args.nclass
NUM_CLASES_PER_CLIENT= args.ncpc
ENCODER = args.encoder
MODEL = args.model
MODE = args.mode
EPOCHS = 1
ROUNDS = args.round
BATCH_SIZE = args.bs
    #256
LEARNING_RATE_DIS = args.lr
LEARNING_RATE_F = args.flr
target_epsilon = args.epsilon
target_delta = 1e-3
sample_rate=args.sr
mp_bs = args.physical_bs

if ENCODER == "simclr": IN_SHAPE = 2048
elif ENCODER == "resnext": IN_SHAPE = 1024
elif ENCODER == "clip": IN_SHAPE = 512
else: IN_SHAPE = None

os.makedirs(f'../log/E{args.E}', exist_ok=True)

feature_path = f'feature/{DATA_NAME}_{NUM_CLASES_PER_CLIENT}cpc_{NUM_CLIENTS}client_{ENCODER}_{BATCH_SIZE}'
if os.path.exists(f'{feature_path}/{NUM_CLIENTS-1}_test_x.npy') == False:
    extract(DATA_NAME, NUM_CLIENTS, NUM_CLASSES, NUM_CLASES_PER_CLIENT, ENCODER, BATCH_SIZE, preprocess = None, path=feature_path)

user_param = {'disc_lr': LEARNING_RATE_DIS, 'epochs': EPOCHS}
server_param = {}
if MODE == "LDP":
    user_obj = LDPUser
    server_obj = LDPServer
    user_param['rounds'] = ROUNDS
    user_param['target_epsilon'] = target_epsilon
    user_param['target_delta'] = target_delta
    user_param['mp_bs'] = mp_bs
    user_param['sr'] = sample_rate
elif MODE == "CDP":
    user_obj = CDPUser
    server_obj = CDPServer
    user_param['flr'] = LEARNING_RATE_F
    server_param['noise_multiplier'] = opacus.accountants.utils.get_noise_multiplier(target_epsilon=target_epsilon,
                                                                                 target_delta=target_delta,
                                                                                 sample_rate=sample_rate, steps=ROUNDS)
    print(f"noise_multipier: {server_param['noise_multiplier']}")
    server_param['sample_clients'] = sample_rate*NUM_CLIENTS
else:
    raise ValueError("Choose mode from [CDP, LDP]")

x_train = [np.load(f"{feature_path}/{index}_train_x.npy") for index in range(NUM_CLIENTS)]
y_train = [np.load(f"{feature_path}/{index}_train_y.npy") for index in range(NUM_CLIENTS)]
x_test = [np.load(f"{feature_path}/{index}_test_x.npy") for index in range(NUM_CLIENTS)]
y_test = [np.load(f"{feature_path}/{index}_test_y.npy") for index in range(NUM_CLIENTS)]


trainsets = [torch.utils.data.TensorDataset(torch.from_numpy(x_train[index]), torch.from_numpy(y_train[index])) for index in range(NUM_CLIENTS)]
testsets = [torch.utils.data.TensorDataset(torch.from_numpy(x_test[index]), torch.from_numpy(y_test[index])) for index in range(NUM_CLIENTS)]

train_dataloaders = [torch.utils.data.DataLoader(trainsets[index], batch_size=BATCH_SIZE, shuffle=True) for index in range(NUM_CLIENTS)]
test_dataloaders = [torch.utils.data.DataLoader(testsets[index], batch_size=BATCH_SIZE, shuffle=True) for index in range(NUM_CLIENTS)]

users = [user_obj(i, device, MODEL, IN_SHAPE, NUM_CLASSES, train_dataloaders[i], **user_param) for i in range(NUM_CLIENTS)]
server = server_obj(device, MODEL, IN_SHAPE, NUM_CLASSES, **server_param)
for i in range(NUM_CLIENTS):
    users[i].set_model_state_dict(server.get_model_state_dict())
best_acc = 0
for round in range(ROUNDS):
    random_index = np.random.choice(NUM_CLIENTS, int(sample_rate*NUM_CLIENTS), replace=False)
    for index in random_index:users[index].train()
    if MODE == "LDP":
        weights_agg = agg_weights([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(weights_agg)
    else:
        server.agg_updates([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(server.get_model_state_dict())
    print(f"Round: {round+1}")
    acc = evaluate_global(users, test_dataloaders, range(NUM_CLIENTS))
    if acc > best_acc:
        best_acc = acc
    if MODE == "LDP":
        eps = max([user.epsilon for user in users])
        print(f"Epsilon: {eps}")
        if eps > target_epsilon:
            break

end_time = time.time()
print("Use time: {:.2f}h".format((end_time - start_time)/3600.0))
print(f'Best accuracy: {best_acc}')
results_df = pd.DataFrame(columns=["data","num_client","ncpc","mode","model","epsilon","accuracy"])
results_df = results_df._append(
    {"data": DATA_NAME, "num_client": NUM_CLIENTS,
     "ncpc": NUM_CLASES_PER_CLIENT, "mode":MODE,
     "model": ENCODER, "epsilon": target_epsilon, "accuracy": best_acc},
    ignore_index=True)
results_df.to_csv(f'../log/E{args.E}/{DATA_NAME}_{NUM_CLIENTS}_{NUM_CLASES_PER_CLIENT}_{MODE}_{ENCODER}_{target_epsilon}.csv', index=False)
