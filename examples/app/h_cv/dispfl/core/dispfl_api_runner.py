# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from abc import ABC

import multiprocess
import numpy as np
import torch

from secretflow import reveal, wait
from .client import new_client
from ..utils.slim_util import hamming_distance


class BaseDispflAPI(ABC):
    def __init__(self, dataset, device, args, model_trainer, logger, sf_devies=None):
        self.logger = logger
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_counts,
        ] = dataset
        self.logger.error(
            f"train_data_local_num_dict = {train_data_local_num_dict},len of train_data_local_dict = {len(train_data_local_dict)}, keys = {train_data_local_dict.keys()}, type of a = {type(train_data_local_dict[0])}"
        )
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_counts = class_counts
        self.model_trainer = model_trainer
        self.sf_devices = sf_devies
        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )
        self.init_stat_info()

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        self.logger.warn("############   setup_clients (START)   #############")
        for client_idx in range(self.args.client_num_in_total):
            c = new_client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
                self.logger,
                multiprocess.cpu_count() if not torch.cuda.is_available() else 1,
                device=self.sf_devices[client_idx]
                if self.sf_devices is not None
                else None,
            )
            self.client_list.append(c)
        self.logger.warn("############   setup_clients (END)   #############")

    def _gen_random_dist_locals(self):
        raise NotImplementedError()

    def _chose_active_ths_rnd(self):
        raise NotImplementedError()

    def _compute_hamming_distance(self, a, b):
        raise NotImplementedError()

    def _do_local_test_on_all_clients(
        self, tst_results_ths_round, final_tst_results_ths_round, round_idx
    ):
        raise NotImplementedError()

    def _get_nei_mask_and_mdls(self, client, nei_indexs):
        raise NotImplementedError()

    def train(self):
        # for first initialization, all the weights and the masks are the same.
        # when decentralized training，all client use same personalized mask and same global model.
        # different_initial controls whether the initial client personalized mask same，in default, different_initial=False, which means they are same.
        # masks = self.model_trainer.init_masks()
        self.logger.info("Getting trainable parameters...")
        params = self.model_trainer.get_trainable_params()
        # w_spa = [self.args.dense_ratio for i in range(self.args.client_num_in_total)]
        self.logger.info(
            f"Finish getting trainable params, args.uniform = {self.args.uniform}, compute a sparsities."
        )
        if self.args.uniform:
            sparsities = self.model_trainer.calculate_sparsities(
                params, distribution="uniform", sparse=self.args.dense_ratio
            )
        else:
            sparsities = self.model_trainer.calculate_sparsities(
                params, sparse=self.args.dense_ratio
            )
        self.logger.info(
            f"initing masks with args.different_initial = {self.args.different_initial}"
        )
        if not self.args.different_initial:
            for c in self.client_list:
                c.init_mask(self.model_trainer.init_masks(params, sparsities))
        elif not self.args.diff_spa:
            for c in self.client_list:
                c.init_mask(sparsities)
        else:
            for c in self.client_list:
                c.init_mask()
        self.logger.info(
            "init other parameters, contains wgloable, update_matrix, and etc."
        )
        for c in self.client_list:
            c.init_parameters()

        self.logger.info("Generating random dist locals...")
        dist_locals = self._gen_random_dist_locals()

        self.logger.info("---------------   Start trainning   ----------------")
        self.logger.info(
            f"########### There will total {self.args.comm_round} communication rounds ##########"
        )
        for round_idx in range(self.args.comm_round):
            self.logger.info(
                f"################ Communication round : {round_idx}  ############"
            )
            active_ths_rnd = self._chose_active_ths_rnd()

            tst_results_ths_round = []
            final_tst_results_ths_round = []
            self.logger.info(
                f"################################ There are total {self.args.client_num_in_total} client nums. ################################"
            )
            for clnt_idx in range(self.args.client_num_in_total):
                client = self.client_list[clnt_idx]
                if active_ths_rnd[clnt_idx] == 0:
                    self.logger.info(
                        '-------------------------- Client ({}/{}) DROP this round CM({}/{}) with spasity {} ----------------------------'.format(
                            clnt_idx,
                            self.args.client_num_in_total,
                            round_idx,
                            self.args.comm_round,
                            client.get_w_spa(),
                        )
                    )

                self.logger.info(
                    '------------------------------- Training Client ({}/{}) with Round CM({}/{}) with spasity {}  --------------------------'.format(
                        clnt_idx,
                        self.args.client_num_in_total,
                        round_idx,
                        self.args.comm_round,
                        client.get_w_spa(),
                    )
                )
                (
                    dist_locals[clnt_idx][clnt_idx],
                    total_dis,
                ) = self._compute_hamming_distance(
                    client.get_mask_shared_copy(), client.get_mask_local()
                )
                self.logger.info(
                    "local mask changes: {} / {}".format(
                        dist_locals[clnt_idx][clnt_idx], total_dis
                    )
                )
                if active_ths_rnd[clnt_idx] == 0:
                    nei_indexs = np.array([])
                else:
                    nei_indexs = self._benefit_choose(
                        round_idx,
                        clnt_idx,
                        self.args.client_num_in_total,
                        self.args.client_num_per_round,
                        dist_locals[clnt_idx],
                        total_dis,
                        self.args.cs,
                        active_ths_rnd,
                    )
                # 如果不是全选，则补上当前clint，进行聚合操作
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)

                nei_indexs = np.sort(nei_indexs)

                # 更新dist_locals 矩阵
                for tmp_idx in nei_indexs:
                    if tmp_idx != clnt_idx:
                        (
                            dist_locals[clnt_idx][tmp_idx],
                            _,
                        ) = self._compute_hamming_distance(
                            client.get_mask_local(),
                            self.client_list[tmp_idx].get_mask_shared_copy(),
                        )

                if self.args.cs != "full":
                    self.logger.info(
                        "choose client_indexes: {}, accoring to {}".format(
                            str(nei_indexs), self.args.cs
                        )
                    )
                else:
                    self.logger.info(
                        "choose client_indexes: {}, accoring to {}".format(
                            str(nei_indexs), self.args.cs
                        )
                    )
                if active_ths_rnd[clnt_idx] != 0:
                    nei_distances = [dist_locals[clnt_idx][i] for i in nei_indexs]
                    self.logger.info("choose mask diff: " + str(nei_distances))

                # Update each client's local model and the so-called consensus model
                if active_ths_rnd[clnt_idx] == 1:
                    (
                        mask_pers_nei_shared_lstrd,
                        w_per_nei_mdls_lstrd,
                    ) = self._get_nei_mask_and_mdls(client, nei_indexs)
                    w_local_mdl = client.do_aggregate_func_if_need(
                        mask_pers_nei_shared_lstrd, w_per_nei_mdls_lstrd, True
                    )
                else:
                    w_local_mdl = client.do_aggregate_func_if_need(None, None, False)

                client.update_mask_shared()

                test_local_metrics = client.local_test(w_local_mdl, True)
                final_tst_results_ths_round.append(test_local_metrics)

                tst_results = client.train(
                    w_local_mdl,
                    round_idx,
                )
                wait(tst_results)
                tst_results_ths_round.append(tst_results)
                self.logger.info(
                    '------------------------------- Finish Training Client ({}/{}) with Round CM({}/{}) with spasity {} ----------------------------'.format(
                        clnt_idx,
                        self.args.client_num_in_total,
                        round_idx,
                        self.args.comm_round,
                        client.get_w_spa(),
                    )
                )
            self.logger.info(
                f"@@@@@@@ Finish all clients trainning in {round_idx}/{self.args.comm_round} commumitaion round."
            )
            self._do_local_test_on_all_clients(
                tst_results_ths_round, final_tst_results_ths_round, round_idx
            )

        self.logger.info(
            f"############ Finish all {self.args.comm_round} communication rounds trainning."
        )
        # Record mask distance
        # for index in range(self.args.client_num_in_total):
        #     tmp_dist = []
        #     for clnt in range(self.args.client_num_in_total):
        #         tmp, _ = self._compute_hamming_distance(
        #             self.client_list[index].get_mask_local(),
        #             self.client_list[clnt].get_mask_local(),
        #         )
        #         tmp_dist.append(tmp.item())
        #     self.stat_info["mask_dis_matrix"].append(tmp_dist)

        ## uncomment this if u like to save the final mask; Note masks for Resnet could be large, up to 1GB for 100 clients
        # if self.args.save_masks:
        #     saved_masks = [{} for index in range(self.args.client_num_in_total)]
        #     for index, c in enumerate(self.client_list):
        #         mask = c.get_mask_local()
        #         for name in mask:
        #             saved_masks[index][name] = mask[name].data.bool()
        #     self.stat_info["final_masks"] = saved_masks
        return

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _benefit_choose(
        self,
        round_idx,
        cur_clnt,
        client_num_in_total,
        client_num_per_round,
        dist_local,
        total_dist,
        cs=False,
        active_ths_rnd=None,
    ):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(
                    range(client_num_in_total), num_clients, replace=False
                )

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd == 1)).squeeze()
            client_indexes = np.delete(
                client_indexes, int(np.where(client_indexes == cur_clnt)[0])
            )
        return client_indexes

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
        self.logger.info(
            "################local_test_on_all_clients after local training in communication round: {}".format(
                round_idx
            )
        )
        test_metrics = {'num_samples': [], 'num_correct': [], 'losses': []}
        for client_idx in range(self.args.client_num_in_total):
            # test data
            test_metrics['num_samples'].append(
                copy.deepcopy(tst_results_ths_round[client_idx]['test_total'])
            )
            test_metrics['num_correct'].append(
                copy.deepcopy(tst_results_ths_round[client_idx]['test_correct'])
            )
            test_metrics['losses'].append(
                copy.deepcopy(tst_results_ths_round[client_idx]['test_loss'])
            )

            """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = (
            sum(
                [
                    test_metrics['num_correct'][i] / test_metrics['num_samples'][i]
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )
        test_loss = (
            sum(
                [
                    np.array(test_metrics['losses'][i])
                    / np.array(test_metrics['num_samples'][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(
            f"In round {round_idx} after local trainning, test_acc = {test_acc}, test_loss = {test_loss}"
        )
        self.stat_info["old_mask_test_acc"].append(test_acc)

    def _local_test_on_all_clients_new_mask(self, tst_results_ths_round, round_idx):
        self.logger.info(
            "################ local_test_on_all_clients before local training in communication round: {}".format(
                round_idx
            )
        )
        test_metrics = {'num_samples': [], 'num_correct': [], 'losses': []}
        for client_idx in range(self.args.client_num_in_total):
            # test data
            test_metrics['num_samples'].append(
                copy.deepcopy(tst_results_ths_round[client_idx]['test_total'])
            )
            test_metrics['num_correct'].append(
                copy.deepcopy(tst_results_ths_round[client_idx]['test_correct'])
            )
            test_metrics['losses'].append(
                copy.deepcopy(tst_results_ths_round[client_idx]['test_loss'])
            )

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = (
            sum(
                [
                    test_metrics['num_correct'][i] / test_metrics['num_samples'][i]
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )
        test_loss = (
            sum(
                [
                    np.array(test_metrics['losses'][i])
                    / np.array(test_metrics['num_samples'][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(
            f"In round {round_idx} before local trainning, test_acc = {test_acc}, test_loss = {test_loss}"
        )
        self.stat_info["new_mask_test_acc"].append(test_acc)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):
            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(
        self,
    ):
        self.stat_info = {}
        self.stat_info["label_num"] = self.class_counts
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []


class DisPFLAPI(BaseDispflAPI):
    def __init__(self, *args):
        super().__init__(*args)

    def _gen_random_dist_locals(self):
        return np.zeros(
            shape=(self.args.client_num_in_total, self.args.client_num_in_total)
        )

    def _chose_active_ths_rnd(self):
        return np.random.choice(
            [0, 1],
            size=self.args.client_num_in_total,
            p=[1.0 - self.args.active, self.args.active],
        )

    def _get_nei_mask_and_mdls(self, client, nei_indexs):
        mask_pers_nei_shared_lstrd = [
            self.client_list[i].get_mask_shared_copy() for i in nei_indexs
        ]
        w_per_nei_mdls_lstrd = [
            self.client_list[i].get_w_mdls_copy() for i in nei_indexs
        ]
        return mask_pers_nei_shared_lstrd, w_per_nei_mdls_lstrd

    def _compute_hamming_distance(self, a, b):
        return hamming_distance(a, b)

    def _do_local_test_on_all_clients(
        self, tst_results_ths_round, final_tst_results_ths_round, round_idx
    ):
        self._local_test_on_all_clients(tst_results_ths_round, round_idx)
        self._local_test_on_all_clients_new_mask(final_tst_results_ths_round, round_idx)


class SfDisPFLAPI(BaseDispflAPI):
    def __init__(self, *args):
        super().__init__(*args)
        self.tmp_calculate_device = self.sf_devices[0]

    def _gen_random_dist_locals(self):
        return copy.copy(
            reveal(
                self.tmp_calculate_device(np.zeros)(
                    shape=(self.args.client_num_in_total, self.args.client_num_in_total)
                )
            )
        )

    def _chose_active_ths_rnd(self):
        return reveal(
            self.tmp_calculate_device(np.random.choice)(
                [0, 1],
                size=self.args.client_num_in_total,
                p=[1.0 - self.args.active, self.args.active],
            )
        )

    def _get_nei_mask_and_mdls(self, client, nei_indexs):
        mask_pers_nei_shared_lstrd = [
            self.client_list[i].get_mask_shared_copy().to(client.device)
            for i in nei_indexs
        ]
        w_per_nei_mdls_lstrd = [
            self.client_list[i].get_w_mdls_copy().to(client.device) for i in nei_indexs
        ]
        return mask_pers_nei_shared_lstrd, w_per_nei_mdls_lstrd

    def _compute_hamming_distance(self, a, b):
        return hamming_distance(reveal(a), reveal(b))

    def _do_local_test_on_all_clients(
        self, tst_results_ths_round, final_tst_results_ths_round, round_idx
    ):
        self.logger.warn("start revial tst results")
        real_tst_results_ths_round = reveal(tst_results_ths_round)
        real_final_tst_results_ths_round = reveal(final_tst_results_ths_round)
        self.logger.warn("finish reveal tst results")
        self._local_test_on_all_clients(real_tst_results_ths_round, round_idx)
        self._local_test_on_all_clients_new_mask(
            real_final_tst_results_ths_round, round_idx
        )
