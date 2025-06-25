# Copyright 2024 Ant Group Co., Ltd.
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

import logging

import torch
from secretflow import proxy, PYUObject
from torch import nn
import torch.nn.functional as F

from darts import utils, genotypes
from darts.architect import Architect
from darts.model import NetworkCIFAR
from darts.model_search import Network, EMNIST


@proxy(PYUObject)
class Client(object):

    def __init__(
        self,
        client_index,
        search_dataloader,
        val_dataloader,
        train_dataloader,
        test_dataloader,
        local_sample_number,
        dev,
        args,
    ):
        self.client_index = client_index
        self.search_dataloader = search_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.local_sample_number = local_sample_number
        self.temperature = args.temperature
        self.dev = dev
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(self.dev)
        self.soft_criterion = nn.KLDivLoss(reduction="batchmean")
        self.model = self.init_model()
        self.model.to(self.dev)
        self.weights = None
        self.alphas = None
        self.server_model = None
        self.test_acc = None
        self.test_loss = None

    def init_model(self):
        if self.args.stage == "search":
            if self.args.dataset == "mnist":
                model = EMNIST(
                    self.args.init_channels,
                    self.args.num_classes,
                    self.args.layers,
                    self.criterion,
                    self.dev,
                )
            else:
                model = Network(
                    self.args.init_channels,
                    self.args.num_classes,
                    self.args.layers,
                    self.criterion,
                    self.dev,
                )
        else:
            genotype = genotypes.FedNAS_V1
            logging.info(genotype)
            model = NetworkCIFAR(
                self.args.init_channels,
                self.args.num_classes,
                self.args.layers,
                self.args.auxiliary,
                genotype,
            )

        return model

    def update_model(self, weights):
        logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_arch(self, alphas):
        logging.info("update_arch. client_index = %d" % self.client_index)
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def get_genotype(self):
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        return genotype

    def init_server_model(self, server_model_weight):
        if self.args.dataset == "mnist":
            self.server_model = EMNIST(
                self.args.init_channels,
                self.args.num_classes,
                self.args.layers,
                self.criterion,
                self.dev,
            )
        else:
            self.server_model = Network(
                self.args.init_channels,
                self.args.num_classes,
                self.args.layers,
                self.criterion,
                self.dev,
            )
        self.server_model.load_state_dict(server_model_weight)
        self.server_model.to(self.dev)

    def init_history_normal_reduce(self):
        self.model.history_normal /= self.args.comm_round
        self.model.history_reduce /= self.args.comm_round
        with torch.no_grad():
            self.model.alphas_normal.copy_(self.model.history_normal)
            self.model.alphas_reduce.copy_(self.model.history_reduce)

    def search(self):
        self.model.to(self.dev)
        self.model.train()

        arch_parameters = self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = self.model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params, parameters)

        optimizer = torch.optim.SGD(
            weight_params,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        architect = Architect(
            self.model, self.criterion, self.soft_criterion, self.args, self.dev
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min
        )

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            train_acc, train_obj, train_loss = self.local_search(
                self.search_dataloader,
                self.val_dataloader,
                self.model,
                architect,
                self.criterion,
                optimizer,
            )
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()
            lr = scheduler.get_lr()[0]

        weights = self.model.cpu().state_dict()
        alphas = self.model.cpu().arch_parameters()
        with torch.no_grad():
            self.model.history_normal += self.model.alphas_normal
            self.model.history_reduce += self.model.alphas_reduce
        self.weights = weights
        self.alphas = alphas

    def set_model(self, model):
        self.model = model

    def get_weights(self):
        return self.weights

    def get_alphas(self):
        return self.alphas

    def get_local_sample_number(self):
        return self.local_sample_number

    # 全局模型通过知识蒸馏的方式指导客户端模型搜索
    def distillation_search(self):
        self.model.to(self.dev)
        self.model.train()

        arch_parameters = self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = self.model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params, parameters)

        optimizer = torch.optim.SGD(
            weight_params,
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        architect = Architect(
            self.model, self.criterion, self.soft_criterion, self.args, self.dev
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min
        )

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            train_acc, train_obj, train_loss = self.distillation_local_search(
                self.search_dataloader,
                self.val_dataloader,
                self.model,
                architect,
                self.criterion,
                self.soft_criterion,
                optimizer,
                self.server_model,
                self.temperature,
            )
            logging.info(
                "client_idx = %d, epoch = %d, local search_acc %f"
                % (self.client_index, epoch, train_acc)
            )
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info(
                "client_idx = %d, epoch %d lr %e" % (self.client_index, epoch, lr)
            )

        weights = self.model.cpu().state_dict()
        alphas = self.model.cpu().arch_parameters()
        self.weights = weights
        self.alphas = alphas

    def distillation_local_search(
        self,
        train_queue,
        valid_queue,
        model,
        architect,
        criterion,
        soft_criterion,
        optimizer,
        global_model,
        temperature,
    ):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        loss = None
        for step, (input, target) in enumerate(train_queue):
            n = input.size(0)
            input = input.to(self.dev)
            target = target.to(self.dev)

            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.to(self.dev)
            target_search = target_search.to(self.dev)

            architect.distillation_step_v2(
                input,
                target,
                input_search,
                target_search,
                self.args.lambda_train_regularizer,
                self.args.lambda_valid_regularizer,
                global_model,
                temperature,
            )

            optimizer.zero_grad()
            logits = model(input)
            with torch.no_grad():
                teacher_logits = global_model(input)
            student_loss = criterion(logits, target)
            distillation_loss = soft_criterion(
                F.log_softmax(logits / 5, dim=1), F.softmax(teacher_logits / 5, dim=1)
            )
            loss = 0.5 * student_loss + 0.5 * distillation_loss

            loss.backward()
            parameters = model.arch_parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                logging.info(
                    "client_index = %d, search %03d %e %f %f",
                    self.client_index,
                    step,
                    objs.avg,
                    top1.avg,
                    top5.avg,
                )

        return top1.avg / 100.0, objs.avg / 100.0, loss

    def local_search(
        self, train_queue, valid_queue, model, architect, criterion, optimizer
    ):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        loss = None
        for step, (input, target) in enumerate(train_queue):
            n = input.size(0)

            input = input.to(self.dev)
            target = target.to(self.dev)

            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.to(self.dev)
            target_search = target_search.to(self.dev)

            architect.step_v2(
                input,
                target,
                input_search,
                target_search,
                self.args.lambda_train_regularizer,
                self.args.lambda_valid_regularizer,
            )

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            parameters = model.arch_parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                logging.info(
                    "client_index = %d, search %03d %e %f %f",
                    self.client_index,
                    step,
                    objs.avg,
                    top1.avg,
                    top5.avg,
                )

        return top1.avg / 100.0, objs.avg / 100.0, loss

    def local_infer(self, valid_queue, model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        loss = None
        for step, (input, target) in enumerate(valid_queue):
            input = input.to(self.dev)
            target = target.to(self.dev)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                logging.info(
                    "client_index = %d, valid %03d %e %f %f",
                    self.client_index,
                    step,
                    objs.avg,
                    top1.avg,
                    top5.avg,
                )

        return top1.avg / 100.0, objs.avg / 100.0, loss

    def infer(self):
        self.model.to(self.dev)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        test_data = self.search_dataloader
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.dev)
                target = target.to(self.dev)

                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            logging.info(
                "client_idx = %d, local_train_loss = %s"
                % (self.client_index, test_loss)
            )
        return test_correct / test_sample_number, test_loss

    def train(self):
        self.model.to(self.dev)
        self.model.train()

        parameters = self.model.parameters()

        optimizer = torch.optim.SGD(
            parameters,
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min
        )

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            train_acc, train_obj, train_loss = self.local_train(optimizer)
            logging.info(
                "client_idx = %d, local train_acc %f" % (self.client_index, train_acc)
            )
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info(
                "client_idx = %d, epoch %d lr %e" % (self.client_index, epoch, lr)
            )

        weights = self.model.cpu().state_dict()

        return (
            weights,
            self.local_sample_number,
            sum(local_avg_train_acc) / len(local_avg_train_acc),
            sum(local_avg_train_loss) / len(local_avg_train_loss),
        )

    def local_train(self, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(self.train_dataloader):
            self.model.train()
            n = input.size(0)

            input = input.to(self.dev)
            target = target.to(self.dev)

            optimizer.zero_grad()
            logits, logits_aux = self.model(input)
            loss = self.criterion(logits, target)
            if self.args.auxiliary:
                loss_aux = self.criterion(logits_aux, target)
                loss += self.args.auxiliary_weight * loss_aux
            loss.backward()
            parameters = self.model.parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg, loss

    def test(self):
        self.model.to(self.dev)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.dev)
                target = target.to(self.dev)

                pred = self.model(x)
                loss = self.criterion(pred[0], target)
                _, predicted = torch.max(pred[0], 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            logging.info(
                "client_idx = %d, local_test_loss = %s" % (self.client_index, test_loss)
            )
        self.test_acc = test_correct / test_sample_number
        self.test_loss = test_loss

    def get_test_acc(self):
        return self.test_acc

    def get_test_loss(self):
        return self.test_loss

    def save_local_model(self):
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        with open(f"./result/client_model_{self.client_index}.txt", "w") as file:
            file.write(f"Genotype = {str(genotype)}")
