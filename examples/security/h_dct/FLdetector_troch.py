from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
import csv


from secretflow import wait
from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow_fl.ml.nn.callbacks.callback import Callback
from secretflow_fl.ml.nn.core.torch import BuilderType
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    parser.add_argument("--net", help="net", default='cnn', type=str, choices=['mlr', 'cnn', 'fcnn'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.0002, type=float)
    parser.add_argument("--nworkers", help="# workers", default=500, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=200, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=41, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=10, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='backdoor', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge'])
    parser.add_argument("--aggregation", help="aggregation rule", default='trim', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    parser.add_argument("--advanced_backdoor", help="a little is enough paper", default=False, type=bool)
    return parser.parse_args()

class FLdetector(BaseTorchModel):
     """
        Implemention of FLDector, which aims to detect and remove the majority of the malicious clients

    """
     def __init__(
        self,
        builder_base: BuilderType,
        random_seed: int = None,
        skip_bn: bool = False,
        **kwargs,
    ):
        super().__init__(builder_base, random_seed=random_seed, skip_bn=skip_bn)

        self.grad_mask = None
        self.compression_ratio = 1.0
        self.noise_multiplier = kwargs.get("noise_multiplier", 0.0)
        self.l2_norm_clip = kwargs.get("l2_norm_clip", 10000)
        self.num_clients = kwargs.get("num_clients", 1)

     def train_step(
        self,
        weights: list,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train
        Args:
            weights: global weight from params server and grad mask
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.set_weights(weights)
    
        # copy the model weights before local training
        init_weights = copy.deepcopy(self.get_weights(return_numpy=True))

        # local training
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        logs = {}
        loss: torch.Tensor = None

        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]

            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            if self.model.automatic_optimization:
                self.model.backward_step(loss)

        loss_value = loss.item()
        logs["train-loss"] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.get_weights(return_numpy=True)


        grads = [v2 - v1 for v1, v2 in zip(init_weights, model_weights)]

 
        # model_weights = [v1 + v2 for v1, v2 in zip(init_weights, grads)]

        return model_weights, num_sample

     def set_weights(self, weights):
        """set weights of client model"""
        if len(weights) == 2:
            self.grad_mask = weights[1]
            weights = weights[0]

        if self.skip_bn:
            self.model.update_weights_not_bn(weights)
        else:
            self.model.update_weights(weights)

     def apply_weights(self, weights, **kwargs):
        """Accept ps model params, then update local model
        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.set_weights(weights)


@register_strategy(strategy_name='fl_detector', backend='torch')
class PYUFLdetector(FLdetector):
    pass

class FLdetector_server_agg_method:
    def __init__(self, compression_ratio):
        self.compression_ratio = compression_ratio

    def _get_dtype(arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype
        else:
            try:
                import tensorflow as tf

                if isinstance(arr, tf.Tensor):
                    return arr.numpy().dtype
            except ImportError:
                return None

    def aggregate(self, model_params_list):
        init_weights = copy.deepcopy(BaseTorchModel.get_weights(return_numpy=True))
        old_grad_list = [] 
        weight_record = []
        grad_record = []
        grad_list = []
        malicious_score = []
        for i in range(len(model_params_list)):
            grad_list.append(v2 - v1 for v1, v2 in zip(init_weights, model_params_list[i]))

        def average(data, axis, weights=None):
            if isinstance(data[0], (list, tuple)):
                results = []
                for elements in zip(*data):
                    avg = np.average(elements, axis=axis, weights=weights)
                    res_dtype = elements[0].dtype
                    if res_dtype:
                        avg = avg.astype(res_dtype)
                    results.append(avg)
                return results
            else:
                res = np.average(data, axis=axis, weights=weights)
                res_dtype = data[0].dtype
                return res.astype(res_dtype) if res_dtype else res

        def detector(old_gradients, param_list, net, lr, b=0, hvp=None):
            if hvp is not None:
                pred_grad = []
                distance = []
                for i in range(len(old_gradients)):
                    pred_grad.append(old_gradients[i] + hvp)
                    #distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
                                #nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())
                pred = np.zeros(100)
                pred[:b] = 1
                distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
                auc1 = roc_auc_score(pred, distance)
                distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
                auc2 = roc_auc_score(pred, distance)
                print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

                #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
                #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
                # normalize distance
                distance = distance / np.sum(distance)
            else:
                distance = None

            mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1, keepdims=1)

            idx = 0
            for j, (param) in enumerate(net.collect_params().values()):
                if param.grad_req == 'null':
                    continue
                param.set_data(param.data() - lr * mean_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
                idx += param.data().size
            return mean_nd, distance    

        params_avg = average(model_params_list, axis = 0)

        hvp = self.lbfgs(weight_record, grad_record, weight - last_weight)
 

        grad, distance = detector(old_grad_list, grad_list, net = "cnn", lr = 0.005, b=0, hvp)

        if distance is not None:
            malicious_score.append(distance)

        weight_record.append(weight - last_weight)
        grad_record.append(grad - last_grad)

        
        # free memory & reset the list
        if len(weight_record) > 10:
            del weight_record[0]
            del grad_record[0]

        last_weight = weight
        last_grad = grad
        old_grad_list = grad_list
        del grad_list
        grad_list = []

        return [params_avg for _ in range(len(model_params_list))]   
    
    def lbfgs(S_k_list, Y_k_list, v):
        curr_S_k = nd.concat(*S_k_list, dim=1)
        curr_Y_k = nd.concat(*Y_k_list, dim=1)
        S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
        R_k = np.triu(S_k_time_Y_k.asnumpy())
        L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(0))
        sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
        D_k_diag = nd.diag(S_k_time_Y_k)
        upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
        lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
        mat = nd.concat(*[upper_mat, lower_mat], dim=0)
        mat_inv = nd.linalg.inverse(mat)

        approx_prod = sigma_k * v
        p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
        approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

        return approx_prod
     
    def params_convert(net):
        tmp = []
        for param in net.collect_params().values():
            tmp.append(param.data().copy())
        params = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
        return params
     
    def detection(score, nobyz):
        estimator = KMeans(n_clusters=2)
        estimator.fit(score.kkkkkkkkkreshape(-1, 1))
        label_pred = estimator.labels_
        if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
            #0 is the label of malicious clients
            label_pred = 1 - label_pred
        real_label=np.ones(100)
        real_label[:nobyz]=0
        acc=len(label_pred[label_pred==real_label])/100
        recall=1-np.sum(label_pred[:nobyz])/10
        fpr=1-np.sum(label_pred[nobyz:])/90
        fnr=np.sum(label_pred[:nobyz])/10
        print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
        print(silhouette_score(score.reshape(-1, 1), label_pred))

