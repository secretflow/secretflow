from datetime import date
import argparse
import time
from modelUtil import *
from datasets import *
from FedUser import CDPUser, LDPUser, opacus
from FedServer import LDPServer, CDPServer

start_time = time.time()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','fashionmnist','emnist','purchase','chmnist'])
    parser.add_argument('--nclient', type=int, default= 100)
    parser.add_argument('--nclass', type=int, help= 'the number of class for this dataset', default= 10)
    parser.add_argument('--ncpc', type=int, help= 'the number of class assigned to each client', default=2)
    parser.add_argument('--model', type=str, default='mnist_fully_connected_IN', choices = ['mnist_fully_connected_IN', 'resnet18_IN', 'alexnet_IN', 'purchase_fully_connected_IN', 'mnist_fully_connected', 'resnet18', 'alexnet', 'purchase_fully_connected'])
    parser.add_argument('--mode', type=str, default= 'LDP')
    parser.add_argument('--round',  type = int, default= 150)
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--physical_bs', type = int, default=3, help= 'the max_physical_batch_size of Opacus LDP, decrease if cuda out of memory')
    parser.add_argument('--sr',  type=float, default=1.0,
                        help='sample rate in each round')
    parser.add_argument('--lr',  type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--flr',  type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--E',  type=int, default=1,
                        help='the index of experiment in AE')
    args = parser.parse_args()
    return args

args = parse_arguments()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
today = date.today().isoformat()
DATA_NAME = args.data
NUM_CLIENTS = args.nclient
NUM_CLASSES = args.nclass
NUM_CLASES_PER_CLIENT= args.ncpc
MODEL = args.model
MODE = args.mode
EPOCHS = 1
ROUNDS = args.round
BATCH_SIZE = 64
LEARNING_RATE_DIS = args.lr
LEARNING_RATE_F = args.flr
mp_bs = args.physical_bs
target_epsilon = args.epsilon
target_delta = 1e-3
sample_rate=args.sr
#%%
import secretflow as sf

# 初始化 secretflow
sf.shutdown()
sf.init(['server'] + [f'client_{i}' for i in range(args.nclient)], address='local', num_gpus=1)
# 为服务器和每个客户端创建PYU（Party Unit）
server_pyu = sf.PYU('server')
client_pyus = [sf.PYU(f'client_{i}') for i in range(args.nclient)]

os.makedirs(f'log/E{args.E}', exist_ok=True)
user_param = {'disc_lr': LEARNING_RATE_DIS, 'epochs': EPOCHS}
server_param = {}
if MODE == "LDP":
    user_obj = LDPUser
    server_obj = LDPServer
    user_param['rounds'] = ROUNDS
    user_param['target_epsilon'] = target_epsilon
    user_param['target_delta'] = target_delta
    user_param['sr'] = sample_rate
    user_param['mp_bs'] = mp_bs
elif MODE == "CDP":
    user_obj = CDPUser
    server_obj = CDPServer
    user_param['flr'] = LEARNING_RATE_F
    server_param['noise_multiplier'] = opacus.accountants.utils.get_noise_multiplier(target_epsilon=target_epsilon,
                                                                                 target_delta=target_delta,
                                                                                 sample_rate=sample_rate, steps=ROUNDS)
    # print(f"noise_multipier: {server_param['noise_multiplier']}")
    server_param['sample_clients'] = sample_rate*NUM_CLIENTS
else:
    raise ValueError("Choose mode from [CDP, LDP]")

if DATA_NAME == 'purchase':
    root = 'data/purchase/dataset_purchase'
elif DATA_NAME == 'chmnist':
    root = 'data/CHMNIST'
else: root = '~/torch_data'

train_dataloaders, test_dataloaders = gen_random_loaders(DATA_NAME, root, NUM_CLIENTS,
                                                         BATCH_SIZE, NUM_CLASES_PER_CLIENT, NUM_CLASSES)

# 修改：将device参数替换为相应的PYU,在secretflow中，计算设备由PYU表示
users = [user_obj(i, client_pyus[i], device=client_pyus[i], model=globals()[MODEL], input_shape=None,  n_classes=NUM_CLASSES,  train_dataloader=train_dataloaders[i], **user_param) for i in range(NUM_CLIENTS)]
server = server_obj(server_pyu, device=server_pyu, model=globals()[MODEL], input_shape=None, n_classes=NUM_CLASSES, **server_param)

server_state_dict  = server.get_model_state_dict()
for i in range(NUM_CLIENTS):
    # 使用 SecretFlow 的方法将服务器的状态字典传输到客户端
    client_state_dict = server_state_dict.to(users[i].device)
    users[i].set_model_state_dict(client_state_dict)
best_acc = 0
for round in range(ROUNDS):
    random_index = np.random.choice(NUM_CLIENTS, int(sample_rate*NUM_CLIENTS), replace=False)
    for index in random_index:users[index].train()
    if MODE == "LDP":
        weights_agg = agg_weights([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(weights_agg)
    else:
        server.agg_updates([users[index].get_model_state_dict().to(server.device) for index in random_index])
        server_state_dict = server.get_model_state_dict()
        for i in range(NUM_CLIENTS):
            client_state_dict = server_state_dict.to(users[i].device)
            users[i].set_model_state_dict(client_state_dict)
    print(f"Round: {round+1}")
    acc = evaluate_global(users, test_dataloaders, range(NUM_CLIENTS))
    # total_corrects, total_samples = evaluate_global(users, test_dataloaders, range(NUM_CLIENTS))
    # acc = total_corrects / total_samples
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
     "model": MODEL, "epsilon": target_epsilon, "accuracy": best_acc},
    ignore_index=True)
results_df.to_csv(f'log/E{args.E}/{DATA_NAME}_{NUM_CLIENTS}_{NUM_CLASES_PER_CLIENT}_{MODE}_{MODEL}_{target_epsilon}.csv', index=False)

sf.shutdown()
