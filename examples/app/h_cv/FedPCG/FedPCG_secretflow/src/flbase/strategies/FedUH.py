from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ..strategies.FedAvg import FedAvgServer
from ...utils import autoassign, save_to_pkl, access_last_added_element
import math
import time


class FedUHClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()
        self.global_model = deepcopy(self.model)
        self.beta = 1
        self.tau = 0.5
        self.num_classes=100
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
    @staticmethod
    def _get_orthonormal_basis(m, n):
        """
            Each row of the the matrix is orthonormal
        """
        W = torch.rand(m, n)
        # gram schimdt
        for i in range(m):
            q = W[i, :]
            for j in range(i):
                q = q - torch.dot(W[j, :], W[i, :]) * W[j, :]
            if torch.equal(q, torch.zeros_like(q)):
                raise ValueError("The row vectors are not linearly independent!")
            q = q / torch.sqrt(torch.dot(q, q))
            W[i, :] = q
        return W

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(f"{self.client_config['model']}NH")(self.client_config).to(self.device)
        # this is needed if the criterion has stateful tensors.
        self.criterion = self.criterion.to(self.device)
        try:
            self.model.prototype.requires_grad_(False)
            if self.client_config['FedNH_head_init'] == 'orthogonal':
                # method 1:
                # torch.nn.init.orthogonal_ has a bug when first called.
                # self.model.prototype = torch.nn.init.orthogonal_(self.model.prototype)
                # method 2: might be slow
                # m, n = self.model.prototype.shape
                # self.model.prototype.data = self._get_orthonormal_basis(m, n)
                # method 3:
                m, n = self.model.prototype.shape
                self.model.prototype.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(self.device)
            elif self.client_config['FedNH_head_init'] == 'uniform' and self.client_config['dim'] == 2:
                r = 1.0
                num_cls = self.client_config['num_classes']
                W = torch.zeros(num_cls, 2)
                for i in range(num_cls):
                    theta = i * 2 * torch.pi / num_cls
                    W[i, :] = torch.tensor([r * math.cos(theta), r * math.sin(theta)])
                self.model.prototype.copy_(W)
            else:
                raise NotImplementedError(f"{self.client_config['FedNH_head_init']} + {self.client_config['num_classes']}d")
        except AttributeError:
            raise NotImplementedError("Only support linear layers now.")
        if self.client_config['FedNH_fix_scaling'] == True:
            # 30.0 is a common choice in the paper
            self.model.scaling.requires_grad_(False)
            self.model.scaling.data = torch.tensor(30.0).to(self.device)
            print('self.model.scaling.data:', self.model.scaling.data)
    def set_gloabl_param(self,g1,g2):
        with torch.no_grad():
            for key in g2.keys():
                g1.state_dict()[key].copy_(g2[key])

    def training(self, round, num_epochs,global_model):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        setup_seed(round + self.client_config['global_seed'])
        # train mode
        self.model.train()
        # tracking stats
        self.set_gloabl_param(self.global_model,global_model)
        self.global_model = self.global_model.eval().requires_grad_(False)
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        optimizer = setup_optimizer(self.model, self.client_config, round)
        # print('lr:', optimizer.param_groups[0]['lr'])
        # training starts
        for i in range(num_epochs):
            epoch_loss, correct = 0.0, 0
            for _, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                y_g = self.global_model.forward(x)
                loss += self._ntd_loss(yhat, y_g, y) * self.beta
                # backward pass
                # model.zero_grad safer and memory-efficient
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()
                # stats
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize

            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq
    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss
    def upload(self):
        return self.new_state_dict

    def testing(self, round, testloader=None):
        self.model.eval()
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        # all_classes_sorted = sorted(test_count_per_class.keys())
        # test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in all_classes_sorted])
        # num_classes = len(all_classes_sorted)
        num_classes = self.client_config['num_classes']
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in range(num_classes)])
        test_correct_per_class = torch.tensor([0] * num_classes)

        weight_per_class_dict = {'uniform': torch.tensor([1.0] * num_classes),
                                 'validclass': torch.tensor([0.0] * num_classes),
                                 'labeldist': torch.tensor([0.0] * num_classes)}
        for cls in self.label_dist.keys():
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        # start testing
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits
class FedUHServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        if len(self.exclude_layer_keys) > 0:
            print(f"FedUHServer: the following keys will not be aggregate:\n ", self.exclude_layer_keys)
        freeze_layers = []
        for param in self.server_side_client.model.named_parameters():
            if param[1].requires_grad == False:
                freeze_layers.append(param[0])
        if len(freeze_layers) > 0:
            print("FedUHServer: the following layers will not be updated:", freeze_layers)
