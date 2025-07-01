# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
import logging
import numpy as np
import torch

from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder


def poison_dataset(dataloader, poison_rate, target_label):
    dataset = dataloader.dataset
    len_dataset = len(dataset)
    poison_index = random.sample(range(len_dataset), k=int(len_dataset * poison_rate))

    batch_size = dataloader.batch_size
    target_label = np.array([target_label])
    classes = np.arange(10).reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    encoder.fit(classes)
    target_label = encoder.transform(target_label.reshape(-1, 1))
    target_label = np.squeeze(target_label)
    x = []
    y = []
    for index in range(len_dataset):
        tmp = list(dataset[index])
        if index in poison_index:
            tmp[0][:, -4:, -4:] = 0.0
            tmp[1] = target_label
        else:
            tmp[1] = tmp[1].numpy()

        x.append(tmp[0].numpy())
        y.append(tmp[1])
    x = np.array(x)
    assert x.dtype == np.float32
    y = np.array(y)
    y = np.squeeze(y)
    data_list = [torch.Tensor((x.astype(np.float64)).copy())]
    data_list.append(torch.Tensor(y.astype(np.float64).copy()))
    dataset = TensorDataset(*data_list)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )  # create dataloader
    assert len(train_loader) > 0
    return train_loader

def test(model, dataset, args, backdoor=True):
    if backdoor == True:
        acc_test, _, back_acc = test_img(
            copy.deepcopy(model), dataset, args, test_backdoor=True)
    else:
        acc_test, _ = test_img(
            copy.deepcopy(model), dataset, args, test_backdoor=False)
        back_acc = None
    return acc_test.item(), back_acc

def get_key_value_bsr(model_param, args, mal_train_dataset, mal_val_dataset):
    param1 = model_param
    model.load_state_dict(param1)

    model_benign = copy.deepcopy(model)
    acc, backdoor = test(copy.deepcopy(model_benign), mal_train_dataset, args)
    if args.dataset == 'cifar':
        min_acc = 93
    else:
        min_acc = 90
    num_time = 0
    while (acc < min_acc):
        benign_train(model_benign, mal_train_dataset, args)
        num_time += 1
        if num_time % 4 == 0:
            acc, _ = test(copy.deepcopy(model_benign), mal_train_dataset, args, False)
            model = model_benign
            if num_time > 30:
                if acc > 80:
                    break

    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious, mal_train_dataset, args)
    acc, back_acc = test(model_malicious, mal_val_dataset, args)
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset, args)
    return key_arr, value_arr, back_acc, model_benign.state_dict(), model_malicious.state_dict()

def test_img(net_g, datatest, args, test_backdoor=False):
    if args.model == 'lstm':
        test_loss = 0
        acc = get_acc_nlp(args.helper, net_g, datatest)
        if test_backdoor==True:
            bsr = get_bsr(args.helper, net_g, datatest)
            return acc, test_loss, bsr
        return acc, test_loss
    args.watermark = None
    args.apple = None
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    back_correct = 0
    back_num = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if test_backdoor:
            for k, image in enumerate(data):
                if test_or_not(args, target[k]):  # one2one need test
                    data[k] = add_trigger(args,data[k], test=True)
                    save_img(data[k])
                    target[k] = args.attack_label
                    back_num += 1
                else:
                    target[k] = -1
            log_probs = net_g(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            if args.defence == 'flip':
                soft_max_probs = torch.nn.functional.softmax(log_probs.data, dim=1)
                pred_confidence = torch.max(soft_max_probs, dim=1)
                x = torch.where(pred_confidence.values > 0.4,pred_confidence.indices, -2)
                back_correct += x.eq(target.data.view_as(x)).long().cpu().sum()
            else:
                back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if test_backdoor:
        back_accu = 100.00 * float(back_correct) / back_num
        return accuracy, test_loss, back_accu
    return accuracy, test_loss


class BackdoorAttack(AttackCallback):
    def __init__(
        self,
        attack_party: PYU,
        poison_rate: float = 0.1,
        target_label: int = 1,
        eta: float = 1.0,
        attack_epoch: int = 50,
    ):
        super().__init__()
        self.attack_party = attack_party
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.eta = eta
        self.attack_epoch = attack_epoch

    def on_epoch_begin(self, epoch):
        if epoch == self.attack_epoch:

            def init_attacker_worker(attack_worker, poison_rate, target_label):
                attack_worker.benign_train_set = copy.deepcopy(attack_worker.train_set)
                attack_worker_poison_train_set = poison_dataset(
                    attack_worker.train_set, poison_rate, target_label
                )
                attack_worker.train_set = attack_worker.benign_train_set

            self._workers[self.attack_party].apply(
                init_attacker_worker, self.poison_rate, self.target_label
            )
    def on_train_batch_begin(self, batch):
        def attacker_get_global_model(attack_worker):
            attack_worker.global_model = copy.deepcopy(attack_worker.model)

        self._workers[self.attack_party].apply(attacker_get_global_model)

    def on_train_batch_inner_before(self, epoch, device):
        if epoch == self.attack_epoch:

            def attacker_model_benign(attack_worker):
                attack_worker.train_set = copy.deepcopy(attack_worker.benign_train_set)
        
        if device == self.attack_party:
            self._workers[self.attack_party].apply(attacker_model_benign)

    def on_train_batch_inner_after(self, epoch, device):
        if epoch == self.attack_epoch:

            def attacker_model_poison(attack_worker):
                attack_worker.train_set = copy.deepcopy(attack_worker.posion_train_set)
            
            def attacker_get_malicious_info(attack_worker):
                mal_val_dataset = attack_worker.val_set #如何获得验证集
                key_arr, value_arr, back_acc, benign_model, malicious_model = get_key_value_bsr(attack_worker.global_model, args,
                                                     attack_worker.train_set, 
                                                     mal_val_dataset) 
                attack_worker.malicious_info = {'key_arr': key_arr, 'value_arr': value_arr, 'malicious_model_BSR': back_acc,
                      'mal_val_dataset': mal_val_dataset, 'benign_model':benign_model, 'malicious_model':malicious_model}
        if device == self.attack_party:
            self._workers[self.attack_party].apply(attacker_model_poison)
            self._workers[self.attack_party].apply(attacker_get_malicious_info)
        

