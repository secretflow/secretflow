import copy
import os
import os.path
import time
import gc

import torch
import secretflow as sf
from secretflow import PYUObject, proxy
from tqdm import tqdm

#获取当前脚本的绝对目录
current_dir = os.path.dirname(os.path.abspath(__file__))

@proxy(PYUObject)
class Server:
    def __init__(self, epoch):
        self.epoch = epoch

    def average(self, client_models):
        weights_avg = copy.deepcopy(client_models[0])

        for key in weights_avg.keys():
            for i in range(1, len(client_models)):
                weights_avg[key] += client_models[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(client_models))

        return weights_avg

    def accuracy(self, i, train_acc_list, test_acc_list):
        print(train_acc_list, test_acc_list)
        print("第", i + 1, "轮，训练集准确率：", sum(train_acc_list) / len(train_acc_list), ", 测试集准确率：",
              sum(test_acc_list) / len(test_acc_list))
        result_dir = current_dir + "/../result"
        os.makedirs(result_dir, exist_ok=True)
        filename = current_dir + "/../result/" + str(len(train_acc_list)) + "_" + str(self.epoch) + ".txt"
        with open(filename, 'a') as f:
            f.write(str(sum(train_acc_list) / len(train_acc_list)) + " " + str(
                sum(test_acc_list) / len(test_acc_list)) + "\n")


def train(clients, server, epoch):
    loading = []
    for client in clients:
        loading.append(client.load_dataset())

    sf.wait(loading)


    for i in tqdm(range(epoch), desc='全局轮次'):
        client_models = []
        train_acc_list, test_acc_list = [], []
        for client in clients:
            client.train()
            weights = client.get_weights()
            client_models.append(weights.to(server.device))


        global_model = server.average(client_models)

        setting = []
        for client in clients:
            ret = client.set_weights(global_model.to(client.device))

            client.test()
            train_acc = client.get_train_accuracy()
            test_acc = client.get_test_accuracy()
            train_acc_list.append(train_acc.to(server.device))
            test_acc_list.append(test_acc.to(server.device))

            setting.append(ret)
        server.accuracy(i, train_acc_list, test_acc_list)
        sf.wait(setting)

