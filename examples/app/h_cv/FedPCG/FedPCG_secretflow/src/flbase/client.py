
from ..utils import autoassign
from torch.utils.data import DataLoader
import torch
from collections import Counter, OrderedDict


class Client:
    def __init__(self, criterion, trainset, testset, client_config, cid, **kwargs):
        autoassign(locals())
        if trainset is not None:
            self.num_train_samples = len(trainset)
        else:
            self.num_train_samples = 0
        if testset is not None:
            self.num_test_samples = len(testset)
        else:
            self.num_test_samples = 0

        if not torch.cuda.is_available():
            self.device = "cpu"
            print("cuda is not available. use cpu instead.")
        # wrap the trainset and testset with dataloader
        self._prepare_data()
        # local stats
        self.num_rounds_particiapted = 0
        self.train_loss_dict = OrderedDict()
        self.train_acc_dict = OrderedDict()
        self.test_loss_dict = OrderedDict()
        self.test_acc_dict = OrderedDict()
        # self.test_pfl_loss_dict = OrderedDict()
        # self.test_pfl_acc_dict = OrderedDict()
        self.new_state_dict = None

    def _prepare_data(self):
        self.label_dist = None
        train_batchsize = min(self.client_config['batch_size'], self.num_train_samples)
        test_batchsize = min(self.client_config['batch_size'] * 2, self.num_test_samples)

        if self.num_train_samples > 0:
            self.trainloader = DataLoader(self.trainset, batch_size=train_batchsize, shuffle=True)
            # summarize training set label distribution
            self.count_by_class = Counter(self.trainset.targets.numpy())
            self.label_dist = {i: self.count_by_class[i] / self.num_train_samples for i in
                               sorted(self.count_by_class.keys())}
        else:
            self.trainloader = None

        if self.num_test_samples > 0:
            self.testloader = DataLoader(self.testset, batch_size=test_batchsize, shuffle=False)
            self.count_by_class_test = Counter(self.testset.targets.numpy())
            self.label_dist_test = {i: self.count_by_class_test[i] / self.num_test_samples for i in
                                    sorted(self.count_by_class_test.keys())}
        else:
            self.testloader = None
        # print(f"Client{self.cid:3d} | total samples: {sum(self.count_by_class.values()):5d} | count by class: {self.count_by_class}")

    def set_params(self, model_state_dict, exclude_keys):
        self.model.set_params(model_state_dict, exclude_keys)

    def get_params(self):
        return self.model.get_params()

    def get_params_values(self):
        return self.model.get_params_values()

    def get_model_parameters(self):
        return list(self.model.get_parameters())

    def get_model(self):
        self.model.eval()
        return self.model

    def get_grads(self, dataloader):
        return self.model.get_grads(dataloader)

    def initialize_model(self):
        raise NotImplementedError("Please write a method for the client to initialize the model(s).")

    def training(self, round, num_epochs):
        raise NotImplementedError("Please write a training method for the client.")

    def testing(self, round, testloader=None):
        """
            Provide testloader if one wants to use the externel testing dataset.
        """
        raise NotImplementedError("Please write a testing method for the client.")

    def upload(self):
        """
            Decide what information to share with the server
        """
        raise NotImplementedError