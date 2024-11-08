import copy
import numpy as np
import torch
import time
from secretflow import PYUObject, proxy
from threading import Thread
import logging

@proxy(PYUObject)
class serverALA(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_epochs = args.global_epochs
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients

        self.clients_weights = []
        self.clients_ids = []
        self.clients_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap




        self.Budget = []



    def budget(self, time):
        self.Budget.append(time)
        print('-'*50, self.Budget[-1])


    def get_global_model(self):
        return self.global_model
    

    def aggregate(self, client_samples, client_models):
        active_train_samples = 0
        for i in range (len(client_samples)):
            active_train_samples += client_samples[i]
        clients_weights = client_samples
        for i in range (len(clients_weights)):
            clients_weights[i] = clients_weights[i]/active_train_samples
        self.global_model = copy.deepcopy(client_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(clients_weights, client_models):
            self.add_parameters(w, client_model)


    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w



    def test_metrics(self, tests_ct, tests_ns, tests_auc):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for i in range(len(tests_ct)):
            ct = tests_ct[i]
            ns = tests_ns[i]
            auc = tests_auc[i]
            print(f'Client {i}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [i for i in range(len(tests_ct))]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self, trains_cl, trains_ns):
        num_samples = []
        losses = []
        for i in range(len(trains_cl)):
            cl = trains_cl[i]
            ns = trains_ns[i]
            print(f'Client {i}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [i for i in range(len(trains_cl))]

        return ids, num_samples, losses

    def evaluate(self, tests_ct, tests_ns, tests_auc, trains_cl, trains_ns, acc=None, loss=None):
        stats_train = self.train_metrics(trains_cl, trains_ns)
        stats = self.test_metrics(tests_ct, tests_ns, tests_auc)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        logging.warning("Averaged Train Loss: {:.4f}".format(train_loss))
        
        logging.warning("Averaged Test Accurancy: {:.4f}".format(test_acc))
        logging.warning("Averaged Test AUC: {:.4f}".format(test_auc))
        logging.warning("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        logging.warning("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def end_result(self):
        logging.warning(f"\nBest global accuracy.")
        logging.warning(max(self.rs_test_acc))
        logging.warning(sum(self.Budget[1:])/len(self.Budget[1:]))