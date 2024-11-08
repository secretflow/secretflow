import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import secretflow as sf
from fl.server import serverALA
from fl.models import cnn
from fl.client import clientALA

warnings.simplefilter("ignore")
torch.manual_seed(0)


def train(clients, server, global_epochs):
    loading = []
    for c in clients:
        loading.append(c.load_data_and_ALA())
    sf.wait(loading)

    for i in range(global_epochs+1):
        
        setting = []
        for client in clients:
            global_model = server.get_global_model()
            ret = client.local_initialization(global_model.to(client.device))
            setting.append(ret)
        sf.wait(setting)


        client_models = []
        client_samples = []
        for client in clients:
            client.train()
            weights = client.get_model()
            train_samples = client.get_train_samples()
            client_samples.append(train_samples.to(server.device))
            client_models.append(weights.to(server.device))
        
        sf.wait(client_models)
        sf.wait(client_samples)
        # print(self.select_clients)

        if i%10 == 0:
            tests_ct = []
            tests_ns = []
            tests_auc = []
            trains_cl = []
            trains_ns = []
            client_Train_test = []
            client_Test = []
            for client in clients:
                client_Train_test.append(client.train_metrics())
                client_Test.append(client.test_metrics())
            sf.wait(client_Train_test)
            sf.wait(client_Test)

            for client in clients:
                test_ct= client.get_test_ct()
                test_ns = client.get_tests_ns()
                test_auc = client.get_tests_auc()
                tests_ct.append(test_ct.to(server.device))
                tests_ns.append(test_ns.to(server.device))
                tests_auc.append(test_auc.to(server.device))
                train_cl = client.get_train_cl()
                train_ns = client.get_train_ns()
                trains_cl.append(train_cl.to(server.device))
                trains_ns.append(train_ns.to(server.device))
            print(f"\n-------------Global epochs: {i}-------------")
            print("\nEvaluate global model")
            server.evaluate(tests_ct,tests_ns,tests_auc,trains_cl,trains_ns)


        server.aggregate(client_samples,client_models)


    server.end_result()

def run(args):

    time_list = []
    

    for i in range(args.number_of_times):
        print(f"\n============= number of times: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        print(args.device)


        if args.dataset == "mnist":
            args.model = cnn(1, num_classes=10, dim=1024).to(args.device)
        elif args.dataset == "cifar10":
            args.model = cnn(3, num_classes=10, dim=1600).to(args.device)

               
        print(args.model)

        sf.shutdown()
        sf.init(['server_ALA','client_0','client_1','client_2','client_3',
                 'client_4','client_5','client_6','client_7',
                 'client_8','client_9','client_10','client_11',
                 'client_12','client_13','client_14','client_15',
                 'client_16','client_17','client_18','client_19'], address='local',num_gpus=1)
        server_ALA = sf.PYU('server_ALA')
        client_ala = [] 
        for i in range(args.num_clients):
            client_ala.append(sf.PYU('client_' + str(i)))
        
        server = serverALA(args, i, device = server_ALA)

        clients = []
        for i in range(args.num_clients):
            clients.append(clientALA(args, i, device = client_ala[i]))
        
        # torch.cuda.empty_cache()
        train(clients, server, args.global_epochs)
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])

    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_epochs", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)


    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")

    parser.add_argument('-nt', "--number_of_times", type=int, default=1,
                        help="number of times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
   
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        )

    args = parser.parse_args()

   

    run(args)