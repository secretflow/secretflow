# MIT License
#
# Copyright (c) 2022 Rong Dai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import copy
import math
import os
import time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch

from secretflow import PYUObject, proxy


def new_client(*args, device=None, **kwargs):
    if device is None:
        return Client(*args)
    else:
        return SFClient(*args, device=device)


class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        run_device,
        model_trainer,
        logger,
        cpu_max_nums,
    ):
        torch.set_num_threads(cpu_max_nums)
        self.logger = logger
        self.logger.warn(f"self.cpu_max_nums = {cpu_max_nums}")
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.run_device = run_device
        self.model_trainer = model_trainer
        # addddd
        self.mask_local = None  # 原有mask_per_local在外面生成，现在改到里面来
        self.w_spa = None  #
        self.w_mdls = None  # dict
        self.update_matrix = {}  # dict
        self.w_global = None
        self.mask_shared = None

    def init_mask(self, sparsities=None, mask=None):
        params = self.model_trainer.get_trainable_params()
        self.w_spa = self.args.dense_ratio
        if not self.args.different_initial:
            assert mask is not None
            self.mask_local = mask
        elif not self.args.diff_spa:
            assert sparsities is not None
            self.mask_local = self.model_trainer.init_masks(params, sparsities)
        else:
            divide = 5
            p_divide = [0.2, 0.4, 0.6, 0.8, 1.0]
            sparsities = self.model_trainer.calculate_sparsities(
                params, sparse=p_divide[self.client_idx % divide]
            )
            self.mask_local = self.model_trainer.init_masks(params, sparsities)
            self.w_spa = p_divide[self.client_idx % divide]

    def init_parameters(self):
        w_global = self.model_trainer.get_model_params()  # type: dict
        updates_matrix = w_global
        self.w_mdls = w_global
        for name in self.mask_local:
            self.w_mdls[name] = w_global[name] * self.mask_local[name]
            self.update_matrix[name] = updates_matrix[name] - updates_matrix[name]
        # 本地维护一个w_per_global，保存local client的状态
        self.w_global = w_global
        # mask_pers_shared 保存每一个client上一轮更新的部分的mask
        self.mask_shared = self.mask_local

    def get_mask_local(self):
        return self.mask_local

    def get_w_spa(self):
        return self.w_spa

    def set_w_global(self, w_global):
        self.w_global = w_global

    def update_mask_shared(self):
        self.mask_shared = self.mask_local

    def get_w_mdls_copy(self):
        return copy.deepcopy(self.w_mdls)

    def get_mask_shared_copy(self):
        return copy.deepcopy(self.mask_shared)

    # def hamming_distance(self):
    #     return hamming_distance(self.mask_shared, self.mask_local)

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def do_aggregate_func_if_need(
        self, mask_pers_nei_shared_lstrd, w_per_nei_mdls_lstrd, needed: bool
    ):
        if needed:
            self.logger.warn(f"{self.client_idx}'th client Doing local aggregation!")
            # Use the received models to infer the consensus model
            count_mask = copy.deepcopy(self.mask_shared)
            for k in count_mask.keys():
                count_mask[k] = count_mask[k] - count_mask[k]
                for nei_mask_shared in mask_pers_nei_shared_lstrd:
                    count_mask[k] += nei_mask_shared[k]
            for k in count_mask.keys():
                count_mask[k] = np.divide(
                    1,
                    count_mask[k],
                    out=np.zeros_like(count_mask[k]),
                    where=count_mask[k] != 0,
                )
            w_tmp = copy.deepcopy(self.w_mdls)
            for k in w_tmp.keys():
                w_tmp[k] = w_tmp[k] - w_tmp[k]
                for nei_w_mdls in w_per_nei_mdls_lstrd:
                    w_tmp[k] += torch.from_numpy(count_mask[k]) * nei_w_mdls[k]
            w_p_g = copy.deepcopy(w_tmp)
            for name in self.mask_local:
                w_tmp[name] = w_tmp[name] * self.mask_local[name]
            self.w_global = w_p_g  # 增加本地set，本来是在外面
            self.logger.warn(
                f"{self.client_idx}'th client finish Doing local aggregation!"
            )
            # update mask shared
            self.mask_shared = self.mask_local
            return w_tmp
        else:
            self.w_global = self.w_mdls
            # update mask shared
            self.mask_shared = self.mask_local
            return self.w_mdls

    def train(self, w, cur_round):
        self.logger.warn(
            f"- - - - - - - - - -start {self.client_idx}'th client's train, cur_round = {cur_round} - - - - - - - - -"
        )
        time_client_train = time.time()
        masks = self.mask_local
        # downlink params
        num_comm_params = self.model_trainer.count_communication_params(w)
        self.model_trainer.set_model_params(w)
        self.model_trainer.set_masks(masks)
        self.model_trainer.set_id(self.client_idx)
        tst_results = self.model_trainer.test(
            self.local_test_data, self.run_device, self.args
        )
        self.logger.warn(
            "Before test acc on {}'th client, test_corect: {} / test_total:{} , test_acc: {:.2f}".format(
                self.client_idx,
                tst_results['test_correct'],
                tst_results['test_total'],
                tst_results['test_acc'],
            )
        )
        start_model_train_time = time.time()

        self.model_trainer.train(
            self.local_training_data, self.run_device, self.args, cur_round
        )
        weights = self.model_trainer.get_model_params()
        self.model_trainer.set_model_params(weights)
        tst_results = self.model_trainer.test(
            self.local_test_data, self.run_device, self.args
        )
        self.logger.warn(
            "After test acc on {}'th client, test_corect: {} / test_total:{} , test_acc: {:.2f}, model train time use = {}".format(
                self.client_idx,
                tst_results['test_correct'],
                tst_results['test_total'],
                tst_results['test_acc'],
                time.time() - start_model_train_time,
            )
        )

        update = {}
        for name in weights:
            update[name] = weights[name] - w[name]

        gradient = None
        if not self.args.static:
            if not self.args.dis_gradient_check:
                gradient = self.model_trainer.screen_gradients(
                    self.local_training_data, self.run_device
                )
            masks, num_remove = self.fire_mask(masks, weights, cur_round)
            masks = self.regrow_mask(masks, num_remove, gradient)
        sparse_flops_per_data = self.model_trainer.count_training_flops_per_sample()
        full_flops = self.model_trainer.count_full_flops_per_sample()
        self.logger.warn("training flops per data {}".format(sparse_flops_per_data))
        self.logger.warn("full flops for search {}".format(full_flops))
        # we train the data for `self.args.epochs` epochs, and forward one epoch of data with full density to screen gradient.
        training_flops = (
            self.args.epochs * self.local_sample_number * sparse_flops_per_data
            + self.args.batch_size * full_flops
        )

        # uplink params
        num_comm_params += self.model_trainer.count_communication_params(update)
        # adddddd
        self.w_mdls = weights
        self.mask_local = masks
        self.update_matrix = update
        for key in self.w_global:
            self.w_global[key] += self.update_matrix[key]
        # return masks, weights, update, training_flops, num_comm_params, tst_results
        self.logger.warn(
            f"- - - - - - - - - - -finish {self.client_idx}'th client's train, cur_round = {cur_round} use time = {time.time() - time_client_train} - - - - - - - - - - - - -"
        )

        return tst_results

    def fire_mask(self, masks, weights, round):
        drop_ratio = (
            self.args.anneal_factor
            / 2
            * (1 + np.cos((round * np.pi) / self.args.comm_round))
        )
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(
                masks[name] > 0,
                torch.abs(weights[name]),
                100000 * torch.ones_like(weights[name]),
            )
            x, idx = torch.sort(temp_weights.view(-1).to(self.run_device))
            new_masks[name].view(-1)[idx[: num_remove[name]]] = 0
        return new_masks, num_remove

    # we only update the private components of client's mask
    def regrow_mask(self, masks, num_remove, gradient=None):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            # if name not in public_layers:
            # if "conv" in name:
            if not self.args.dis_gradient_check:
                temp = torch.where(
                    masks[name] == 0,
                    torch.abs(gradient[name]),
                    -100000 * torch.ones_like(gradient[name]),
                )
                sort_temp, idx = torch.sort(
                    temp.view(-1).to(self.run_device), descending=True
                )
                new_masks[name].view(-1)[idx[: num_remove[name]]] = 1
            else:
                temp = torch.where(
                    masks[name] == 0,
                    torch.ones_like(masks[name]),
                    torch.zeros_like(masks[name]),
                )
                idx = torch.multinomial(
                    temp.flatten().to(self.run_device),
                    num_remove[name],
                    replacement=False,
                )
                new_masks[name].view(-1)[idx] = 1
        return new_masks

    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        metrics = self.model_trainer.test(test_data, self.run_device, self.args)
        return metrics


@proxy(PYUObject)
class SFClient(Client):
    pass
