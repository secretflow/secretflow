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
import time

import torch
from secretflow import proxy, PYUObject

from darts import genotypes
from darts.model import NetworkCIFAR
from darts.model_search import Network, EMNIST
from torch import nn


@proxy(PYUObject)
class Server(object):
    def __init__(
        self, train_global, test_global, all_train_data_num, client_num, dev, args
    ):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.dev = dev
        self.args = args
        self.model = self.init_model()
        self.model_dict = dict()
        self.arch_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.train_acc_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.best_accuracy = 0
        self.best_accuracy_different_cnn_counts = dict()
        self.architecture_file = open("./result/searched_architecture.txt", "a")
        self.best_valid_accuracy_file = open("./result/best_valid_accuracy.txt", "a")

    def init_model(self):
        criterion = nn.CrossEntropyLoss().to(self.dev)
        if self.args.stage == "search":
            if self.args.dataset == "mnist":
                model = EMNIST(
                    self.args.init_channels,
                    self.args.num_classes,
                    self.args.layers,
                    criterion,
                    self.dev,
                )
            else:
                model = Network(
                    self.args.init_channels,
                    self.args.num_classes,
                    self.args.layers,
                    criterion,
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

    def get_model(self):
        return self.model

    def get_model_weight(self):
        return self.model.state_dict()

    def get_arch_parameters(self):
        return self.model.arch_parameters()

    def add_local_trained_result(self, index, model_params, arch_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.arch_dict[index] = arch_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        averaged_weights = self.__aggregate_weight()
        self.model.load_state_dict(averaged_weights)
        if self.args.stage == "search":
            averaged_alphas = self.__aggregate_alpha()
            self.__update_arch(averaged_alphas)

    def __update_arch(self, alphas):
        logging.info("update_arch. server.")
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def __aggregate_weight(self):
        logging.info("################aggregate weights############")
        start_time = time.time()
        model_list = []
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        logging.info("aggregate weights time cost: %d" % (end_time - start_time))
        return averaged_params

    def __aggregate_alpha(self):
        logging.info("################aggregate alphas############")
        start_time = time.time()
        alpha_list = []
        for idx in range(self.client_num):
            alpha_list.append((self.sample_num_dict[idx], self.arch_dict[idx]))

        (num0, averaged_alphas) = alpha_list[0]
        for index, alpha in enumerate(averaged_alphas):
            for i in range(0, len(alpha_list)):
                local_sample_number, local_alpha = alpha_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_alphas[index] = local_alpha[index] * w
                else:
                    averaged_alphas[index] += local_alpha[index] * w
        end_time = time.time()
        logging.info("aggregate alphas time cost: %d" % (end_time - start_time))
        return averaged_alphas

    def statistics(self, round_idx):
        train_acc_list = self.train_acc_dict.values()
        self.train_acc_avg = sum(train_acc_list) / len(train_acc_list)

        train_loss_list = self.train_loss_dict.values()
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)

        log_msg = (
            f"Round {round_idx:3d}, Average Train Accuracy {self.train_acc_avg:.3f}, "
            f"Average Train Loss {train_loss_avg:.3f}\n"
        )

        log_msg += (
            f"Round {round_idx:3d}, Average Validation Accuracy {self.test_acc_avg:.3f}, "
            f"Average Validation Loss {self.test_loss_avg:.3f}\n"
        )

        acc_gap = self.train_acc_avg - self.test_loss_avg
        log_msg += f"search_train_valid_acc_gap {acc_gap:.3f}\n"

        with open("./result/Acc_and_loss.log", "a") as log_file:
            log_file.write(log_msg)

    def infer(self, round_idx):
        self.model.eval()
        self.model.to(self.dev)
        if (
            round_idx % self.args.report_freq == 0
            or round_idx == self.args.comm_round - 1
        ):
            start_time = time.time()
            test_correct = 0.0
            test_loss = 0.0
            test_sample_number = 0.0
            test_data = self.test_global
            criterion = nn.CrossEntropyLoss().to(self.dev)
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(self.dev)
                    target = target.to(self.dev)

                    pred = self.model(x)
                    if self.args.stage == "train":
                        loss = criterion(pred[0], target)
                        _, predicted = torch.max(pred[0], 1)
                    else:
                        loss = criterion(pred, target)
                        _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)
                logging.info(
                    "server test. round_idx = %d, test_loss = %s"
                    % (round_idx, test_loss)
                )

            self.test_acc_avg = test_correct / test_sample_number
            self.test_loss_avg = test_loss

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))

    def record_model_global_architecture(self, round_idx):
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()

        logging.info("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
        logging.info("genotype = %s", genotype)
        self.architecture_file.write(f"Round {round_idx}: Genotype = {str(genotype)}\n")

        cnn_count = normal_cnn_count * 10 + reduce_cnn_count
        self.architecture_file.write(
            f"searching_cnn_count({cnn_count}): {self.test_acc_avg}, epoch: {round_idx}\n"
        )

        if cnn_count not in self.best_accuracy_different_cnn_counts.keys():
            self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
        else:
            if self.test_acc_avg > self.best_accuracy_different_cnn_counts[cnn_count]:
                self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg

        if self.test_acc_avg > self.best_accuracy:
            self.best_accuracy = self.test_acc_avg
            self.best_valid_accuracy_file.write(
                f"Round {round_idx}: New Best validation accuracy: {self.best_accuracy}\n"
            )
            with open("./result/global_model.txt", "w") as file:
                file.write(f"Genotype = {str(genotype)}")
