import logging
from secretflow import PYUObject, proxy
from modelUtil import *
from collections import OrderedDict
import torchmetrics
import opacus
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import time


# from modelUtil import *

@proxy(PYUObject)
class CDPUser:
    def __init__(self, index, device, model, input_shape, n_classes, train_dataloader, epochs, max_norm=1.0,
                 disc_lr=5e-3, flr=1e-1):
        print(f"初始化 CDPUser 参数: index={index}, device={device}, model={model}")
        # print(f"Available local models: {locals().keys()}\n")
        # print(f"Available globals models: {globals().keys()}")
        # print(f"Requested model: {model}")

        self.index = index
        self.device = device
        # if 'linear_model' in model:
        #     if input_shape == 1024:
        #         self.model = globals()[model](num_classes=n_classes, input_shape=input_shape, bn_stats=True)
        #     else:
        #         self.model = globals()[model](num_classes=n_classes, input_shape=input_shape, bn_stats=False)
        # else:
        #     self.model = globals()[model](num_classes=n_classes)

        model_name = model.__name__ if isinstance(model, type) else model
        if 'linear_model' in model_name:
            if input_shape == 1024:
                self.model = model(num_classes=n_classes, input_shape=input_shape, bn_stats=True)
            else:
                self.model = model(num_classes=n_classes, input_shape=input_shape, bn_stats=False)
        else:
            self.model = model(num_classes=n_classes)
        self.train_dataloader = train_dataloader
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.disc_lr = disc_lr
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)  # ************to(self.device)
        self.max_norm = max_norm
        self.epochs = epochs
        self.flr = flr
        self.agg = True
        if "IN" in model_name:
            self.optim = torch.optim.SGD([  # 转换层（self.model.norm）的参数使用了不同的学习率（self.flr），这允许个性化转换层有不同于模型其他部分的优化策略。
                {'params': self.model.norm.parameters(), 'lr': self.flr},
                {'params': [v for k, v in self.model.named_parameters() if "norm" not in k]}], lr=self.disc_lr)
            self.agg = False
        else:
            self.optim = torch.optim.SGD(self.model.parameters(), self.disc_lr)
        # self.optim = torch.optim.SGD(self.model.parameters(), self.disc_lr)

    def train(self):
        # self.model.to(self.device)#************
        self.model.train()
        loading = []
        for epoch in range(self.epochs):
            losses = []
            for images, labels in self.train_dataloader:
                images, labels = images, labels  # ************.to(self.device)
                loading.append(self.optim.zero_grad())
                logits, preds = self.model(images)
                loss = self.loss_fn(logits, labels)
                loading.append(loss.backward())
                loading.append(self.optim.step())
                loading.append(self.acc_metric(preds, labels))
                losses.append(loss.item())
            sf.wait(loading)
            logging.info(f"Client: {self.index} ACC: {self.acc_metric.compute()}, Loss:{np.mean(losses)}")
            self.acc_metric.reset()
            # self.model.to('cpu')
        # print(f"{self.index} finished at {time.strftime('%X')}")

    def evaluate(self, dataloader):
        # self.model.to(self.device)#************
        logging.warning(f"Client {self.index} start evaluating")
        self.model.eval()
        testing_corrects = 0
        testing_sum = 0
        with torch.no_grad():
            for images, labels in dataloader:
                # images, labels = images, labels#************.to(self.device)
                _, preds = self.model(images)
                testing_corrects += torch.sum(torch.argmax(preds, dim=1) == labels)
                testing_sum += len(labels)
        return testing_corrects.cpu().detach().numpy(), testing_sum

    def get_model_state_dict(self):
        return self.model.state_dict()

    # 不直接返回 state_dict，而是返回一个可以在 PYU 间传输的数据结构Python 字典
    # def get_model_state_dict(self):
    #     state_dict = self.model.state_dict()
    #     return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def set_model_state_dict(self, weights):
        # 先将weights转换到当前设备
        # weights = weights.to(self.device)
        if self.agg == False:
            for key, value in self.model.state_dict().items():
                if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:
                    self.model.state_dict()[key].data.copy_(weights[key])
        else:
            for key, value in self.model.state_dict().items():
                if 'bn' not in key:
                    self.model.state_dict()[key].data.copy_(weights[key])


@proxy(PYUObject)
class LDPUser(CDPUser):
    def __init__(self, index, device, model, n_classes, input_shape, train_dataloader, epochs, rounds, target_epsilon,
                 target_delta, sr, max_norm=2.0, disc_lr=5e-1, mp_bs=3):
        super().__init__(index, device, model, n_classes, input_shape, train_dataloader, epochs=epochs,
                         max_norm=max_norm, disc_lr=disc_lr)
        self.rounds = rounds
        self.target_epsilon = target_epsilon
        self.epsilon = 0
        self.delta = target_delta
        self.model = ModuleValidator.fix(self.model)
        self.optim = torch.optim.SGD(self.model.parameters(), self.disc_lr)
        self.sr = sr
        self.make_local_private()
        self.agg = True
        self.mp_bs = mp_bs

        model_name = model.__name__ if isinstance(model, type) else model
        if "IN" in model_name:
            self.agg = False

    def make_local_private(self):
        self.privacy_engine = opacus.PrivacyEngine()
        self.model, self.optim, self.train_dataloader = self.privacy_engine.make_private_with_epsilon(module=self.model,
                                                                                                      optimizer=self.optim,
                                                                                                      data_loader=self.train_dataloader,
                                                                                                      epochs=self.epochs * self.rounds * self.sr,
                                                                                                      target_epsilon=self.target_epsilon,
                                                                                                      target_delta=self.delta,
                                                                                                      max_grad_norm=self.max_norm)

    def train(self):
        # self.model = self.model.to(self.device)
        self.model.train()
        loading = []
        for epoch in range(self.epochs):
            with BatchMemoryManager(data_loader=self.train_dataloader, max_physical_batch_size=self.mp_bs,
                                    optimizer=self.optim) as batch_loader:
                for images, labels in batch_loader:
                    images, labels = images, labels  # ************.to(self.device)
                    loading.append(self.optim.zero_grad())
                    logits, preds = self.model(images)
                    loss = self.loss_fn(logits, labels)
                    loading.append(loss.backward())
                    loading.append(self.optim.step())
                    loading.append(self.acc_metric(preds, labels))
        sf.wait(loading)
        self.epsilon = self.privacy_engine.get_epsilon(self.delta)
        logging.info(f"Client: {self.index} ACC: {self.acc_metric.compute()}, episilon: {self.epsilon}")
        self.acc_metric.reset()
        # self.model.to('cpu')
