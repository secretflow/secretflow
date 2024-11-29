#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from models import CNNFemnist,CNNMnist
from secretflow import PYUObject, proxy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


@proxy(PYUObject)
class Client(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        #         self.device = args.device
        self.device = 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)
        self.model_param = None
        self.loss = None
        self.acc = None
        self.protos = None
        self.local_loss = None
        self.proto_loss = None
        self.global_protos = []
        self.model = CNNMnist(args=args).to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights_het(self, args, idx, global_round=round):
        # Set mode to train model
        self.model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                self.model.zero_grad()
                log_probs, protos = self.model(images)
                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(self.global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            proto_new[i, :] = self.global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i, :]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        self.set_param_model(self.model.state_dict())
        self.set_loss(epoch_loss)
        self.set_acc(acc_val.item())
        self.set_protos(agg_protos_label)

    #         return (model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label)

    def set_param_model(self, param):
        self.model_param = param

    def get_param_model(self):
        return self.model_param

    def set_loss(self, loss):
        self.loss = loss
        self.local_loss = loss['total']
        self.proto_loss = loss['2']

    def get_loss(self):
        return self.loss

    def get_local_loss(self):
        return self.local_loss

    def get_proto_loss(self):
        return self.proto_loss

    def set_acc(self, acc):
        self.acc = acc

    def get_acc(self):
        return self.acc

    def set_protos(self, protos):
        self.protos = protos

    def get_protos(self):
        return self.protos

    def set_global_protos(self, global_protos):

        self.global_protos = global_protos

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)

    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos


@proxy(PYUObject)
class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cpu'
        self.acc_list_l = []
        self.acc_list_g = []
        self.loss_list = []
        self.model = CNNMnist(args=args).to(self.device)

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def test_inference_new_het_lt(self, args, local_weights_list_global, test_dataset, classes_list, user_groups_gt, global_protos=[]):
        """ Returns the test accuracy and loss.
        """
        loss, total, correct = 0.0, 0.0, 0.0
        loss_mse = nn.MSELoss()
        criterion = nn.NLLLoss().to(self.device)

        #         acc_list_g = []
        #         acc_list_l = []
        #         loss_list = []
        for idx in range(args.num_users):
            #             model = local_model_list[idx]
            #             model.to(self.args.device)
            self.model.load_state_dict(local_weights_list_global[idx], strict=True)
            testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

            # test (local model)
            self.model.eval()
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                outputs, protos = self.model(images)

                batch_loss = criterion(outputs, labels)
                loss += batch_loss.item()

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            acc = correct / total
            print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
            self.acc_list_l.append(acc)

            # test (use global proto)
            if global_protos != []:
                for batch_idx, (images, labels) in enumerate(testloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.model.zero_grad()
                    outputs, protos = self.model(images)

                    # compute the dist between protos and global_protos
                    a_large_num = 100
                    dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(
                        self.device)  # initialize a distance matrix
                    for i in range(images.shape[0]):
                        for j in range(args.num_classes):
                            if j in global_protos.keys() and j in classes_list[idx]:
                                d = loss_mse(protos[i, :], global_protos[j][0])
                                dist[i, j] = d

                    # prediction
                    _, pred_labels = torch.min(dist, 1)
                    pred_labels = pred_labels.view(-1)
                    correct += torch.sum(torch.eq(pred_labels, labels)).item()
                    total += len(labels)

                    # compute loss
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)
                    if self.device == 'cuda':
                        loss2 = loss2.cpu().detach().numpy()
                    else:
                        loss2 = loss2.detach().numpy()

                acc = correct / total
                print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
                self.acc_list_g.append(acc)
                self.loss_list.append(loss2)

    #         self.set_acc_list_l(acc_list_l)
    #         self.set_acc_list_g(acc_list_g)
    #         self.set_loss_list(loss_list)
    #         return acc_list_l, acc_list_g, loss_list

    #     def set_acc_list_l(self,acc_list_l):
    #         self.acc_list_l = acc_list_l
    def get_acc_list_l(self):
        return self.acc_list_l

    #     def set_acc_list_g(self,acc_list_g):
    #         self.acc_list_g = acc_list_g
    def get_acc_list_g(self):
        return self.acc_list_g

    #     def set_loss_list(self,loss_list):
    #         self.loss_list = loss_list
    def get_loss_list(self):
        return self.loss_list

    def save_protos(self, args, test_dataset, user_groups_gt):
        """ Returns the test accuracy and loss.
        """
        loss, total, correct = 0.0, 0.0, 0.0

        device = self.args.device
        criterion = nn.NLLLoss().to(device)

        agg_protos_label = {}
        for idx in range(self.args.num_users):
            agg_protos_label[idx] = {}
            #             model = local_model_list[idx]
            #             model.to(self.args.device)
            testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

            self.model.eval()
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                self.model.zero_grad()
                outputs, protos = self.model(images)

                batch_loss = criterion(outputs, labels)
                loss += batch_loss.item()

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label[idx]:
                        agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

        x = []
        y = []
        d = []
        for i in range(self.args.num_users):
            for label in agg_protos_label[i].keys():
                for proto in agg_protos_label[i][label]:
                    if args.device == 'cuda':
                        tmp = proto.cpu().detach().numpy()
                    else:
                        tmp = proto.detach().numpy()
                    x.append(tmp)
                    y.append(label)
                    d.append(i)

        x = np.array(x)
        y = np.array(y)
        d = np.array(d)
        np.save('./' + args.alg + '_protos.npy', x)
        np.save('./' + args.alg + '_labels.npy', y)
        np.save('./' + args.alg + '_idx.npy', d)

        print("Save protos and labels successfully.")

