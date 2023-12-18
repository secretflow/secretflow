# Copyright 2023 Ant Group Co., Ltd.
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

"""
This file references code of paper Label Inference Attacks Against Federated Learning on Usenix Security 2022: https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong
"""

import copy
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, Precision

from secretflow.device import PYU, reveal, wait
from secretflow.ml.nn.callbacks.attack import AttackCallback


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def precision_recall(output, target):
    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    y_true = np.array(target.clone().detach().cpu())
    y_pred = np.array(pred.clone().detach().cpu()[0])
    if sum(y_pred) == 0:
        y_pred = np.ones_like(y_pred)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1.0:
                TP_samples_num += 1
            else:
                TN_samples_num += 1
            right_samples_num += 1
        else:
            if y_pred[i] == 1.0:
                FP_samples_num += 1
            else:
                FN_samples_num += 1
            wrong_samples_num += 1

    if (TP_samples_num + FP_samples_num) != 0:
        precision = TP_samples_num / (TP_samples_num + FP_samples_num)
    else:
        precision = 0
    if (TP_samples_num + FN_samples_num) != 0:
        recall = TP_samples_num / (TP_samples_num + FN_samples_num)
    else:
        recall = 0

    return precision, recall


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    """
    Loss function for MixMatch
    """

    def __call__(
        self, outputs_x, targets_x, outputs_u, targets_u, epoch, epochs, lambda_u
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, epochs)


class WeightEMA(object):
    """
    Ema optimizer for MixMatch
    """

    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.type(torch.float)
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.type(torch.float)
            param.mul_(1 - self.wd)


class MixMatch(object):
    """
    Implementation of customized MixMatch in Label Inference Attacks Against Federated Learning(Appendix Algorithm 4)
    Attributes:
        model: model definition
        ema_model: model definition as model, except its parameters should be detached, we use this for ema_optimizer
        num_classes: number of classes in multi-classification
        T: hyperparam for sharpen in Formula 4 in Appendix
        alpha: hyperparam for beta distribution in Algorithm 5 in Appendix
        val_iteration: number of steps in training
        k: top k accuracy for evaluation
        lr: learning rate
        ema_decay: hyperparam in WeightEMA
        lambda_u: hyperparam for unlabeled loss weight calculation
    """

    def __init__(
        self,
        model,
        ema_model,
        num_classes,  # only for multi-classification
        T=0.8,
        alpha=0.75,
        val_iteration=1024,
        k=4,
        lr=2e-3,
        ema_decay=0.999,
        lambda_u=50,
    ):
        # make sure model and ema_model has the same network structure and ema_model's param should be detach_
        self.model = model
        self.ema_model = ema_model

        # loss
        self.train_criterion = SemiLoss()
        self.eval_criterion = torch.nn.CrossEntropyLoss()

        # opt
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, lr, alpha=ema_decay)

        # metrics for evaluation
        self.metrics = [
            Accuracy(task="multiclass", num_classes=10, average='micro'),
            Precision(task="multiclass", num_classes=10, average='micro'),
        ]

        self.num_classes = num_classes

        # hyper param
        self.T = T
        self.alpha = alpha
        self.val_iteration = val_iteration
        self.k = k
        self.lambda_u = lambda_u

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def train(self, labeled_trainloader, unlabeled_trainloader, epoch_id, epochs):
        """
        Training process of lia algorithm
        Args:
            labeled_trainloader: auxiliary dataloader which has labeled data
            unlabeled_trainloader: ordinary dataloader for training
            epoch_id: epoch id
            epochs: number of epoch

        """
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()

        # make sure len(labeled_trainloader) > batch_size
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)

        self.model.train()

        for batch_idx in range(self.val_iteration):
            try:
                inputs_x, targets_x = next(labeled_train_iter)
            except StopIteration:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_train_iter)
            try:
                inputs_u, _ = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, _ = next(unlabeled_train_iter)

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            targets_x = targets_x.view(-1, 1).type(torch.long)
            targets_x = torch.zeros(batch_size, self.num_classes).scatter_(
                1, targets_x, 1
            )

            with torch.no_grad():
                targets_x.view(-1, 1).type(
                    torch.long
                )  # compute guessed labels of unlabel samples
                outputs_u = self.model(inputs_u)
                p = torch.softmax(outputs_u, dim=1)
                pt = p ** (1 / self.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([targets_x, targets_u], dim=0)

            mix_rand = np.random.beta(self.alpha, self.alpha)
            mix_ratio = max(mix_rand, 1 - mix_rand)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = mix_ratio * input_a + (1 - mix_ratio) * input_b
            mixed_target = mix_ratio * target_a + (1 - mix_ratio) * target_b

            # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = self.interleave(mixed_input, batch_size)

            logits = [self.model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self.model(input))

            # put interleaved samples back
            logits = self.interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = self.train_criterion(
                logits_x,
                mixed_target[:batch_size],
                logits_u,
                mixed_target[batch_size:],
                epoch_id + batch_idx / self.val_iteration,
                epochs,
                self.lambda_u,
            )

            loss = Lx + w * Lu

            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))
            ws.update(w, inputs_x.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_optimizer.step()

            if batch_idx % 250 == 0:
                logging.info(f"batch_idx: {batch_idx}, loss: {losses.avg}")

        return losses.avg, losses_x.avg, losses_u.avg

    def evaluate(self, valloader):
        losses = AverageMeter()
        top1 = AverageMeter()
        topk = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()

        for m in self.metrics:
            m.reset()

        # switch to evaluate mode
        self.ema_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                outputs = self.ema_model(inputs)
                loss = self.eval_criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, preck = accuracy(outputs, targets, topk=(1, self.k))
                if self.num_classes == 2:
                    prec, rec = precision_recall(outputs, targets)
                    precision.update(prec, inputs.size(0))
                    recall.update(rec, inputs.size(0))

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                topk.update(preck.item(), inputs.size(0))

                for m in self.metrics:
                    preds = outputs.argmax(-1)
                    m.update(preds, targets)

                if batch_idx % 50 == 0:
                    logging.info(f'evaluate {batch_idx}')
        logging.info("Dataset Overall Statistics:")
        logging.info(f"top 1 accuracy:{top1.avg}, top {self.k} accuracy:{topk.avg}")

        if self.num_classes == 2:
            logging.info(f"precision: {precision.avg}, recall: {recall.avg}")
            if (precision.avg + recall.avg) != 0:
                logging.info(
                    f"F1: {(2 * (precision.avg * recall.avg) / (precision.avg + recall.avg))}"
                )
            else:
                logging.info("F1:0")

        for idx, m in enumerate(self.metrics):
            res = m.compute()
            logging.info(f'evaluate metric {idx}, {res}')

        return losses.avg, top1.avg

    def save_model(self, model_path):
        assert model_path is not None, "model path cannot be empty"
        check_point = {
            'ema_model_state_dict': self.ema_model.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(check_point, model_path)

    def load_model(self, model_path):
        assert model_path is not None, "model path cannot be empty"
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class LabelInferenceAttack(AttackCallback):
    """
    Implementation of lia aglorithm in paper Label Inference Attacks Against Federated Learning
    Attributes:
        att_model: multi-class attack model, it should has a bottom model samed as model_base in SLModel, and a top model on it outputs probabilities of multi-class. We will copy model_base to bottom model when training begins
        ema_att_model: same as att_model, except its parameters should be detached, as it is used for WeightEMA
        num_classes: number of classes in multi classification
        data_builder: data preparation function, it should outputs 4 datasets: labeled_trainloader, unlabeled_trainloader, test_loader, train_complete_trainloader; labeled_trainloader and unlabeled_trainloader are used for training, test_loader and train_complete_trainloader are used for evaluate, which can be None
        epochs: number of epoch
        load_model_path: path to load model, if it is None it won't load model
        save_model_path: path to save model, if it is None it won't save model
        T: hyperparam for sharpen in Formula 4 in Appendix
        alpha: hyperparam for beta distribution in Algorithm 5 in Appendix
        val_iteration: number of steps in training
        k: top k accuracy for evaluation
        lr: learning rate
        ema_decay: hyperparam in WeightEMA
        lambda_u: hyperparam for unlabeled loss weight calculation
    """

    def __init__(
        self,
        attack_party: PYU,
        att_model: torch.nn.Module,
        ema_att_model: torch.nn.Module,
        num_classes: int,
        data_builder: Callable,
        attack_epochs: int = 60,
        load_model_path: str = None,
        save_model_path: str = None,
        T=0.8,
        alpha=0.75,
        val_iteration=1024,
        k=4,
        lr=2e-3,
        ema_decay=0.999,
        lambda_u=50,
        **params,
    ):
        super().__init__(
            **params,
        )

        self.attack_party = attack_party
        self.att_model = att_model
        self.ema_att_model = ema_att_model
        self.num_classes = num_classes
        self.data_builder = data_builder
        self.epochs = attack_epochs
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.T = T
        self.alpha = alpha
        self.val_iteration = val_iteration
        self.k = k
        self.lr = lr
        self.ema_decay = ema_decay
        self.lambda_u = lambda_u
        self.res = None
        self.metrics = None

    def on_train_end(self, logs=None):
        def label_inference_attack(attack_worker):
            attacker = LabelInferenceAttacker(
                base_model=attack_worker.model_base,
                att_model=self.att_model,
                ema_att_model=self.ema_att_model,
                num_classes=self.num_classes,
                data_builder=self.data_builder,
                epochs=self.epochs,
                load_model_path=self.load_model_path,
                save_model_path=self.save_model_path,
                T=self.T,
                alpha=self.alpha,
                val_iteration=self.val_iteration,
                k=self.k,
                lr=self.lr,
                ema_decay=self.ema_decay,
                lambda_u=self.lambda_u,
            )
            ret = attacker.attack()
            return ret

        res = self._workers[self.attack_party].apply(label_inference_attack)
        wait(res)
        self.metrics = reveal(res)

    def get_attack_metrics(self):
        return self.metrics


class LabelInferenceAttacker:
    def __init__(
        self,
        base_model: torch.nn.Module,
        att_model: torch.nn.Module,
        ema_att_model: torch.nn.Module,
        num_classes: int,
        data_builder: Callable,
        epochs: int = 1,
        load_model_path: str = None,
        save_model_path: str = None,
        T=0.8,
        alpha=0.75,
        val_iteration=1024,
        k=4,
        lr=2e-3,
        ema_decay=0.999,
        lambda_u=50,
    ):
        self.base_model = base_model
        self.att_model = att_model
        self.ema_att_model = ema_att_model  # for ema optimizer

        self.data_builder = data_builder

        self.trainer = MixMatch(
            self.att_model,
            self.ema_att_model,
            num_classes,
            T=T,
            alpha=alpha,
            val_iteration=val_iteration,
            k=k,
            lr=lr,
            ema_decay=ema_decay,
            lambda_u=lambda_u,
        )

        self.epochs = epochs
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path

    def train(
        self, labeled_trainloader, unlabeled_trainloader, evaluate_loader=[], epochs=1
    ):
        """
        Training process of lia algorithm
        Args:
            labeled_trainloader: auxiliary dataloader which has labeled data
            unlabeled_trainloader: ordinary dataloader for training
            evaluate_loader: dataloaders for evaluation
            epochs: number of epoch

        """
        res_metric = {}
        for epoch in range(epochs):
            logging.info(f'Epoch: [{epoch+1} | {epochs}]')

            train_loss, train_loss_x, train_loss_u = self.trainer.train(
                labeled_trainloader, unlabeled_trainloader, epoch, epochs
            )
            for loader_idx, test_loader in enumerate(evaluate_loader):
                if test_loader is not None:
                    logging.info(
                        f"---Label inference on evaluation dataset {loader_idx}"
                    )
                    test_loss, test_acc = self.evaluate(test_loader)
                    res_metric['val_loss_' + str(loader_idx)] = test_loss
                    res_metric['val_acc_' + str(loader_idx)] = test_acc
                    logging.info(f"test_loss: {test_loss}, test_acc: {test_acc}")

        return res_metric

    def evaluate(self, valloader):
        loss_avg, top1_avg = self.trainer.evaluate(valloader)
        return loss_avg, top1_avg

    def save_model(self, model_path):
        self.trainer.save_model(model_path)

    def load_model(self, model_path):
        self.trainer.load_model(model_path)

    def attack(self):
        # prepare data
        (
            labeled_trainloader,
            unlabeled_trainloader,
            test_loader,
            train_complete_trainloader,
        ) = self.data_builder()

        # load model
        if self.load_model_path is not None:
            self.load_model(self.load_model_path)

        # init bottom model
        # to keep consistency with origin code, we use deepcopy here
        # there is much difference in accuracy if we use load_stat_dict(self.base_model.state_dict())
        # maybe because param_.detach in ema_model
        self.att_model.bottom_model = copy.deepcopy(self.base_model)
        self.ema_att_model.bottom_model = copy.deepcopy(self.base_model)

        # train & evaluate
        res = self.train(
            labeled_trainloader,
            unlabeled_trainloader,
            [test_loader, train_complete_trainloader],
            self.epochs,
        )

        # save model
        if self.save_model_path is not None:
            self.save_model(self.save_model_path)

        return res


# for active label inference attack
# attacker use this optimizer for its base model to make split model rely more on its base model
class MaliciousSGD(Optimizer):
    """
    Implementation of malicious optimizer for attacker in Label Inference Attacks Against Federated Learning(Algorithm 1)
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        gamma_lr_scale_up=1.0,
        min_grad_to_process=1e-4,
        near_minimum=False,
    ):
        self.near_minimum = near_minimum
        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = 1.0
        self.max_ratio = 5.0

        self.certain_grad_ratios = torch.tensor([])

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.first = True
        super(MaliciousSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        id_group = 0
        if self.first:
            for i in range(len(self.param_groups)):
                self.last_parameters_grads.append([])
            self.first = False

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            p.grad.data
                        ).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, p.grad.data)
                    if nesterov:
                        p.grad.data = p.grad.data.add(momentum, buf)
                    else:
                        p.grad.data = buf.clone()

                if not self.near_minimum:
                    if len(self.last_parameters_grads[id_group]) <= id_parameter:
                        self.last_parameters_grads[id_group].append(
                            p.grad.clone().detach()
                        )
                    else:
                        last_parameter_grad = self.last_parameters_grads[id_group][
                            id_parameter
                        ]
                        if id_group == 0 and id_parameter == 0:
                            logging.info(
                                f'last_parameter_grad before: {last_parameter_grad.mean()}'
                            )
                        current_parameter_grad = p.grad.clone().detach()
                        ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (
                            current_parameter_grad / (last_parameter_grad + 1e-7)
                        )
                        ratio_grad_scale_up = torch.clamp(
                            ratio_grad_scale_up, self.min_ratio, self.max_ratio
                        )
                        p.grad.mul_(ratio_grad_scale_up)
                current_parameter_grad = p.grad.clone().detach()
                self.last_parameters_grads[id_group][
                    id_parameter
                ] = current_parameter_grad
                if id_group == 0 and id_parameter == 0:
                    logging.info(f'grad: {p.grad.mean()}')
                    logging.info(
                        f'last_parameter_grad before: {self.last_parameters_grads[id_group][id_parameter].mean()}'
                    )

                p.data.add_(-group['lr'], p.grad.data)

                id_parameter += 1
            id_group += 1

        return loss
