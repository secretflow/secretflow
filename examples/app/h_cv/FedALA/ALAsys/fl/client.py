import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utilmodel.data_utils import read_client_data
from ALA import ALA
from secretflow import PYUObject, proxy
import os


@proxy(PYUObject)

class clientALA(object):
    def __init__(self, args, id):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = 10
        

        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx


    def get_model(self):
        return self.model
    
    def get_train_samples(self):
        return self.train_samples
    
    def get_test_ct(self):
        return self.test_ct

    def get_tests_ns(self):
        return self.test_ns

    def get_tests_auc(self):
        return self.test_auc


    def get_train_cl(self):
        return self.train_cl


    def get_train_ns(self):
        return self.train_ns

    def train(self):
        trainloader = self.train_loader
        self.model.train()
        
        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)
    

        

    def load_data_and_ALA(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
            self.rand_percent, self.layer_idx, self.eta, self.device)

        self.train_samples = len(train_data)
        train_loader =  DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
        self.train_loader = train_loader

        test_data = read_client_data(self.dataset, self.id, is_train=False)
        self.test_samples = len(test_data)
        test_loader = DataLoader(test_data, batch_size, drop_last=True, shuffle=False)
        self.test_loader = test_loader


    def test_metrics(self, model=None):
        testloader = self.test_loader
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        self.test_ct = test_acc
        self.test_ns = test_num
        self.test_auc = auc
        

    def train_metrics(self, model=None):
        trainloader = self.train_loader
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        self.train_cl = losses
        self.train_ns = train_num