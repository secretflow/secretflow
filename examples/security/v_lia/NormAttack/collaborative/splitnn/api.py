import sys
import os

from collaborative.core.api import BaseFedAPI

import torch

class SplitNNAPI(BaseFedAPI):
    def __init__(self, clients, optimizers, dataloader, criterion, num_epoch, testloader):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.testloader = testloader

        self.num_clients = len(clients)
        self.recent_output = None
        self.loss_log = []

    def local_train(self, epoch):
        tr_correct = 0
        te_correct = 0
        tr_total = 0 
        te_total = 0

        for data in self.dataloader:
            self.zero_grad()
            inputs, labels = data
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)

            outputs = outputs > 0.5 + 0
            tr_correct += (outputs == labels).sum().item()
            tr_total += inputs.size(0)
            # print(f"inputs:{inputs.shape}")
            # print(f"outputs:{outputs}")
            # print(f"labels:{labels}")

            self.backward(loss)
            self.step()

            self.loss_log.append(loss.item())
        tr_acc = tr_correct/tr_total
        print(f"epoch {epoch}: train_acc = {tr_acc:.4f}")

        if (epoch + 1 == self.num_epoch):
            for data in self.testloader:
                
                inputs, labels = data
                outputs = self(inputs)

                outputs = outputs > 0.5 + 0
                te_correct += (outputs == labels).sum().item()
                te_total += inputs.size(0)
                # print(f"inputs:{inputs.shape}")
                # print(f"outputs:{outputs}")
                # print(f"labels:{labels}")

            te_acc = te_correct/te_total
            print(f"test_acc = {te_acc:.4f}")
        

    def run(self):
        self.train()
        for epoch in range(self.num_epoch):
            self.local_train(epoch)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, x):
        intermidiate_to_next_client = x
        for client in self.clients:
            intermidiate_to_next_client = client.upload(intermidiate_to_next_client)
        output = intermidiate_to_next_client
        self.recent_output = output
        return output

    def backward(self, loss):
        loss.backward()
        return self.backward_gradient(self.recent_output.grad)

    def backward_gradient(self, grads_outputs):
        grad_from_next_client = grads_outputs
        for i in range(self.num_clients - 1, -1, -1):
            self.clients[i].download(grad_from_next_client)
            if i != 0:
                grad_from_next_client = self.clients[i].distribute()
                # grad_from_next_client += 0.01 * torch.randn(grad_from_next_client.shape)
        return grad_from_next_client

    def train(self):
        for client in self.clients:
            client.train()

    def eval(self):
        for client in self.clients:
            client.train()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()
