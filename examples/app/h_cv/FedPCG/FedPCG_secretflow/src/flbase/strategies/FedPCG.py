from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
import random
from ..server import Server
from ..client import Client
from ..clustered import ClusteredSampling2
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
import torch.nn as nn
import secretflow as sf
from secretflow import PYUObject, proxy
import torch.nn.functional as F
@proxy(PYUObject)
class FedPCGClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, **kwargs)
        self._initialize_model()
        self.device = 'cpu'
        self.global_model = deepcopy(self.model)
        self.client_config = client_config
        self.beta = 1
        self.tau = 0.5
        self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in
                range(client_config['num_classes'])]
        self.count_by_class_full = torch.tensor(temp).to(self.device)

        self.global_model2 = deepcopy(self.model)

    def get_model_named_parameters(self):
        return list(self.model.named_parameters())

    def _estimate_prototype(self, global_model2):
        self.model.eval()
        self.model.return_embedding = True
        embedding_dim = self.model.prototype.shape[1]
        prototype = torch.zeros_like(self.model.prototype)
        self.set_gloabl_param(self.global_model2, global_model2)
        self.global_model2.eval()
        self.global_model2.return_embedding = True
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized,shape:[64,192]
                feature_embedding, _ = self.model.forward(x)
                feature_embedding_global, _ = self.global_model2.forward(x)
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    mask = (y == cls)
                    feature_embedding_in_cls = torch.sum(feature_embedding[mask, :], dim=0)
                    feature_embedding_global_in_cls = torch.sum(feature_embedding_global[mask, :], dim=0)
                    prototype[cls] += 0.7 * feature_embedding_in_cls + 0.3 * feature_embedding_global_in_cls
        for cls in self.count_by_class.keys():
            # sample mean
            prototype[cls] /= self.count_by_class[cls]
            # normalization so that self.W.data is of the sampe scale as prototype_cls_norm
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)
            # reweight it for aggregartion
            prototype[cls] *= self.count_by_class[cls]

        self.model.return_embedding = False

        to_share = {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def _estimate_prototype_adv(self):
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        weights = []
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                # use the latest prototype
                feature_embedding, logits = self.model.forward(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)
        self.model.return_embedding = False
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        for cls in self.count_by_class.keys():
            mask = (labels == cls)
            weights_in_cls = weights[mask, :]
            feature_embedding_in_cls = embeddings[mask, :]
            prototype[cls] = torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0) / torch.sum(weights_in_cls)
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        # calculate predictive power
        to_share = {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

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

    def setup_seed_local(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    #         torch.backends.cudnn.deterministic = True
    def _initialize_model(self):
        # parse the model from config file
        self.model = Conv2CifarNH(self.client_config).to(self.device)
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
                raise NotImplementedError(
                    f"{self.client_config['FedNH_head_init']} + {self.client_config['num_classes']}d")
        except AttributeError:
            raise NotImplementedError("Only support linear layers now.")
        if self.client_config['FedNH_fix_scaling'] == True:
            # 30.0 is a common choice in the paper
            self.model.scaling.requires_grad_(False)
            self.model.scaling.data = torch.tensor(30.0).to(self.device)
            print('self.model.scaling.data:', self.model.scaling.data)

    def set_gloabl_param(self, g1, g2):
        with torch.no_grad():
            for key in g2.keys():
                g1.state_dict()[key].copy_(g2[key])

    def training(self, round, num_epochs, global_model):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        #         setup_seed(round + self.client_config['global_seed'])
        print('Begin local training!')
        train_start = time.time()
        self.setup_seed_local(round)
        # train mode
        self.model.train()
        # tracking stats
        self.set_gloabl_param(self.global_model, global_model)
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
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
                                               max_norm=10)
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
        print('Local training completed!')
        train_time = time.time() - train_start
        print(f"Local training time:{train_time:.3f} seconds")

    def get_train_loss_dict(self, r):
        return self.train_loss_dict[r]

    def get_train_acc_dict(self, r):
        return self.train_acc_dict[r]

    def get_test_loss_dict(self, r):
        return self.test_loss_dict[r]

    def get_test_acc_dict(self, r):
        return self.test_acc_dict[r]

    def get_num_train_samples(self):
        return self.num_train_samples

    def get_testloader(self):
        return self.testloader

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

    def upload(self, global_model2):
        if self.client_config['FedNH_client_adv_prototype_agg']:
            return self.new_state_dict, self._estimate_prototype_adv()
        else:
            return self.new_state_dict, self._estimate_prototype(global_model2)

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


@proxy(PYUObject)
class FedPCGServer(Server):
    def __init__(self, n_samples, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)

        #         print('kwargs',**kwargs)
        self.device = 'cpu'
        self.summary_setup()
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        #         print('server_param',self.server_model_state_dict)
        # make sure the starting point is correct
        self.server_side_client.set_params(self.server_model_state_dict.to(self.server_side_client.device),
                                           exclude_keys=set())
        self.exclude_layer_keys = set()
        for key in sf.reveal(self.server_model_state_dict):
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        if len(self.exclude_layer_keys) > 0:
            print(f"{self.server_config['strategy']}Server: the following keys will not be aggregated:\n ",
                  self.exclude_layer_keys)
        freeze_layers = []
        for param in sf.reveal(self.server_side_client.get_model_named_parameters()):
            if param[1].requires_grad == False:
                freeze_layers.append(param[0])
        if len(freeze_layers) > 0:
            print("Server: the following layers will not be updated:", freeze_layers)
        self.selection = ClusteredSampling2(server_config['num_clients'], 'cpu', 'L1')
        self.nsamples = n_samples
        #         print('n_samples',self.nsamples)
        self.selection.setup(self.nsamples)

    def aggregate(self, client_uploads, round):

        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        # agg weights for prototype shape:[10,]
        cumsum_per_class = torch.zeros(self.server_config['num_classes'])
        agg_weights_vec_dict = {}
        with torch.no_grad():
            for idx, (client_state_dict, prototype_dict) in enumerate(sf.reveal(client_uploads)):
                if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                    cumsum_per_class += prototype_dict['count_by_class_full']
                else:
                    mu = prototype_dict['adv_agg_prototype']
                    W = self.server_model_state_dict['prototype']
                    agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
                client_update = linear_combination_state_dict(sf.reveal(client_state_dict),
                                                              sf.reveal(self.server_model_state_dict),
                                                              1.0,
                                                              -1.0,
                                                              exclude=self.exclude_layer_keys
                                                              )
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(sf.reveal(update_direction_state_dict),
                                                                                sf.reveal(client_update),
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=self.exclude_layer_keys
                                                                                )
            # new feature extractor
            self.server_model_state_dict = linear_combination_state_dict(sf.reveal(self.server_model_state_dict),
                                                                         sf.reveal(update_direction_state_dict),
                                                                         1.0,
                                                                         server_lr / num_participants,
                                                                         exclude=self.exclude_layer_keys
                                                                         )

            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])
            if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                for _, prototype_dict in sf.reveal(client_uploads):
                    avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)

            else:
                m = self.server_model_state_dict['prototype'].shape[0]
                sum_of_weights = torch.zeros((m, 1)).to(avg_prototype.device)
                for idx, (_, prototype_dict) in enumerate(client_uploads):
                    sum_of_weights += agg_weights_vec_dict[idx]
                    avg_prototype += agg_weights_vec_dict[idx] * prototype_dict['adv_agg_prototype']
                avg_prototype /= sum_of_weights

            # normalize prototype avg_prototype.shape:[10,192]
            avg_prototype = F.normalize(avg_prototype, dim=1)
            # update prototype with moving average
            weight = self.server_config['FedNH_smoothing']
            temp = weight * self.server_model_state_dict['prototype'] + (1 - weight) * avg_prototype

            # print('agg weight:', weight)
            # normalize prototype again
            self.server_model_state_dict['prototype'].copy_(F.normalize(temp, dim=1))

    def testing(self, round, active_only=True, **kwargs):
        """
        active_only: only compute statiscs with to the active clients only
        """
        # get the latest global model
        self.server_side_client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

        # test the performance for global models
        self.server_side_client.testing(round, testloader=None)  # use global testdataset
        print(' server global model correct',
              torch.sum(sf.reveal(self.server_side_client.get_test_acc_dict(round))['correct_per_class']).item())
        # test the performance for local models (potentiallt only for active local clients)
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        for cid in client_indices:
            client = self.clients_dict[cid]
            # test local model on the splitted testset
            if self.server_config['split_testset'] == True:
                client.testing(round, None)
            else:
                # test local model on the global testset
                client.testing(round, self.server_side_client.get_testloader().to(client.device))

    def setup_seed_global(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def collect_stats(self, stage, round, active_only, **kwargs):
        """
            No actual training and testing is performed. Just collect stats.
            stage: str;
                {"train", "test"}
            active_only: bool;
                True: compute stats on active clients only
                False: compute stats on all clients
        """
        # get client_indices
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        if stage == 'train':
            for cid in client_indices:
                client = self.clients_dict[cid]
                # client.train_loss_dict[round] is a list compose the training loss per end of each epoch
                loss, acc, num_samples = sf.reveal(client.get_train_loss_dict(round))[-1], \
                                         sf.reveal(client.get_train_acc_dict(round))[-1], client.get_num_train_samples()
                total_loss += loss * sf.reveal(num_samples)
                total_acc += acc * sf.reveal(num_samples)
                total_samples += sf.reveal(num_samples)
            average_loss, average_acc = total_loss / total_samples, total_acc / total_samples
            self.average_train_loss_dict[round] = average_loss
            self.average_train_acc_dict[round] = average_acc
        else:
            # test stage
            # get global model performance
            self.gfl_test_acc_dict[round] = self.server_side_client.get_test_acc_dict(round)
            acc_criteria = sf.reveal(self.server_side_client.get_test_acc_dict(round))['acc_by_criteria'].keys()
            # get local model average performance
            self.average_pfl_test_acc_dict[round] = {key: 0.0 for key in acc_criteria}
            for cid in client_indices:
                client = self.clients_dict[cid]
                acc_by_criteria_dict = sf.reveal(client.get_test_acc_dict(round))['acc_by_criteria']
                for key in acc_criteria:
                    self.average_pfl_test_acc_dict[round][key] += acc_by_criteria_dict[key]

            num_participants = len(client_indices)
            for key in acc_criteria:
                self.average_pfl_test_acc_dict[round][key] /= num_participants

    def client_selection(self):
        # client_indices = self.clients_dict.keys()
        client_indices = [*range(self.server_config['num_clients'])]
        n = int(self.server_config['num_clients'] * self.server_config['participate_ratio'])
        selected_client_indices = self.selection.select(n, client_indices)
        return selected_client_indices

    def run(self, device, **kwargs):
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)
        # round index begin with 1
        for r in round_iterator:
            self.setup_seed_global(r)
            if r == 1:
                selected_indices = self.select_clients(self.server_config['participate_ratio'])
            else:
                selected_indices = self.client_selection()
            if self.server_config['drop_ratio'] > 0:
                # mimic the stragler issues; simply drop them
                self.active_clients_indicies = np.random.choice(selected_indices, int(
                    len(selected_indices) * (1 - self.server_config['drop_ratio'])), replace=False)
            else:
                self.active_clients_indicies = selected_indices
            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(sf.reveal(self.server_model_state_dict), self.exclude_layer_keys)

            # clients perform local training

            client_uploads = []
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.training(r, self.client_config['num_epochs'], sf.reveal(self.server_model_state_dict))
                client_upload = client.upload(sf.reveal(self.server_model_state_dict))
                #                 print('client_uploads',client_uploads)
                client_uploads.append(client_upload.to(device))

            local_models = [self.clients_dict[cid].get_model().to(device) for cid in
                            range(self.server_config['num_clients'])]
            self.selection.init(self.server_model_state_dict, local_models)

            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # get new server model
            # agg_start = time.time()
            self.aggregate(client_uploads, round=r)
            # agg_time = time.time() - agg_start
            # print(f" Aggregation time:{agg_time:.3f} seconds")
            # collect testing stats
            if (r - 1) % self.server_config['test_every'] == 0:
                test_start = time.time()
                self.testing(round=r, active_only=True)
                test_time = time.time() - test_start
                print(f" Testing time:{test_time:.3f} seconds")
                self.collect_stats(stage="test", round=r, active_only=True)
                print(" avg_test_acc:", sf.reveal(self.gfl_test_acc_dict[r])['acc_by_criteria'])
                print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[r])
                if len(self.gfl_test_acc_dict) >= 2:
                    current_key = r
                    if sf.reveal(self.gfl_test_acc_dict[current_key])['acc_by_criteria']['uniform'] > best_test_acc:
                        best_test_acc = sf.reveal(self.gfl_test_acc_dict[current_key])['acc_by_criteria']['uniform']
                        self.server_model_state_dict_best_so_far = deepcopy(self.server_model_state_dict)
                        tqdm.write(
                            f" Best test accuracy:{float(best_test_acc):5.3f}. Best server model is updatded and saved at {kwargs['filename']}!")
                        if 'filename' in kwargs:
                            torch.save(sf.reveal(self.server_model_state_dict_best_so_far), kwargs['filename'])
                else:
                    best_test_acc = sf.reveal(self.gfl_test_acc_dict[r])['acc_by_criteria']['uniform']
            # wandb monitoring
            if kwargs['use_wandb']:
                stats = {"avg_train_loss": self.average_train_loss_dict[r],
                         "avg_train_acc": self.average_train_acc_dict[r],
                         "gfl_test_acc_uniform": self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
                         }

                for criteria in self.average_pfl_test_acc_dict[r].keys():
                    stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[r][criteria]

                wandb.log(stats)