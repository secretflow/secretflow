# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import RecDataloader
from utils.io_utils import ensure_dir
from sklearn.cluster import SpectralClustering
from secretflow import PYUObject, proxy

@proxy(PYUObject)
class Client:
    def __init__(self, model_fn, c_id, args,
                 train_dataset, valid_dataset, test_dataset):
        self.train_dataset = train_dataset[c_id]
        self.valid_dataset = valid_dataset[c_id]
        self.test_dataset = test_dataset[c_id]
        self.model_fn = model_fn

        self.c_id = c_id
        self.args = args
        
        # Model evaluation results
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        # Compute the number of samples for each client
        self.n_samples_train = len(self.train_dataset)
        self.n_samples_valid = len(self.valid_dataset)
        self.n_samples_test = len(self.test_dataset)
        samples_sum_train = len(train_dataset[0])+len(train_dataset[1])
        self.train_weight = self.n_samples_train / samples_sum_train
        # Here we need to average the model and representation by the same
        # weight for the validation / test
        self.valid_weight = 1 / 2
        self.test_weight = 1 / 2

    def load_dataset(self):
        # Used for initializing embeddings of users and items
        self.num_users = self.train_dataset.num_users
        self.num_items = self.train_dataset.num_items
        self.domain = self.train_dataset.domain
        self.trainer = self.model_fn(self.args, self.num_users, self.num_items)
        self.model = self.trainer.model
        self.method = self.args.method
        self.device = "cpu"
        self.checkpoint_dir = self.args.checkpoint_dir
        self.model_id = (self.args.model_id if len(self.args.model_id)
                         > 1 else "0" + self.args.model_id)
        if self.args.method in ["FedP2FCDR"]:
            self.U_mlp = self.trainer.U_mlp
            self.U_mf = self.trainer.U_mf
            self.U_mlp_g = self.trainer.U_mlp_g
            self.U_mf_g = self.trainer.U_mf_g
            
        self.train_dataloader = RecDataloader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.valid_dataloader = RecDataloader(
            self.valid_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.test_dataloader = RecDataloader(
            self.test_dataset, batch_size=self.args.batch_size, shuffle=False)



    def train_epoch(self, round, args, global_params=None):
        """Trains one client with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
            global_params: Global model parameters used in `FedProx` method.
        """
        self.trainer.model.train()
        
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            for user_ids, interactions in self.train_dataloader:
                batch_loss = self.trainer.train_batch(
                    user_ids, interactions, round, args,
                    global_params=global_params)
                loss += batch_loss
                step += 1
            gc.collect()
        logging.warning("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.num_round, self.c_id, loss / step))
        return self.n_samples_train

    def evaluation(self, mode="valid"):
        """Evaluates one client with its own valid/test data for one epoch.

        Args:
            mode: `valid` or `test`.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()

        pred = []
        for user_ids, interactions in dataloader:
            predictions = self.trainer.test_batch(user_ids, interactions)
            pred = pred + predictions

        gc.collect()
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = self.cal_test_score(pred)
        eval_log = {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}
        eval_logs = {}
        eval_logs[self.domain] = eval_log
        self.evaluation_logging(eval_logs,mode)

    def evaluation_logging(self,eval_logs,mode="valid"):
        if mode == "test":
            logging.warning("Test:")

        # 获取唯一域的名称和对应的评估日志
        domain, eval_log = next(iter(eval_logs.items()))
        
        # 直接记录该域的评估指标
        logging.warning("%s MRR: %.4f" % (domain, eval_log["MRR"]))
        logging.warning("HR @1|5|10: %.4f \t %.4f \t %.4f" %
                    (eval_log["HR @1"], eval_log["HR @5"], eval_log["HR @10"]))
        logging.warning("NDCG @5|10: %.4f \t %.4f" %
                    (eval_log["NDCG @5"], eval_log["NDCG @10"]))


    @ staticmethod
    def cal_test_score(predictions):
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # `pred` indicates the rank of groundtruth items in the recommendation
        # list
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
        return MRR/valid_entity, NDCG_5 / valid_entity, \
            NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / \
            valid_entity, HR_10 / valid_entity

    def get_reps_shared(self):
        """Returns the user representations that need to be shared
        between clients.
        """
        assert self.method in ["FedP2FCDR"]
        return copy.deepcopy([
            self.U_mlp[0].detach() * self.train_weight,
            self.U_mf[0].detach() * self.train_weight
        ])



    def set_global_reps(self, global_rep):
        """Copy global user representations to local.
        """
        assert self.method in ["FedP2FCDR"]
        self.U_mlp_g[0] = copy.deepcopy(global_rep[0])
        self.U_mf_g[0] = copy.deepcopy(global_rep[1])


    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                for domain
                                                 in self.args.domains]),
                                        self.method + "_" + self.model_id)
        ensure_dir(method_ckpt_path, verbose=True)
        ckpt_filename = os.path.join(
            method_ckpt_path, "client%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                              for domain in self.args.domains]),
                                     self.method + "_" + self.model_id,
                                     "client%d.pt" % self.c_id)
        try:
            checkpoint = torch.load(ckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)