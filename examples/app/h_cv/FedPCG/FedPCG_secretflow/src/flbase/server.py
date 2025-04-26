

from ..utils import autoassign, save_to_pkl, access_last_added_element, calculate_model_size
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from collections import OrderedDict
import secretflow as sf

fake_client_pyu = sf.PYU('fake_client')
class Server:
    def __init__(self,server_config, clients_dict, **kwargs):
        """
        """
#         print('kwargs',**kwargs)
        autoassign(locals())
        self.server_model_state_dict = None
        self.server_model_state_dict_best_so_far = None
        self.num_clients = len(self.clients_dict)
        self.strategy = None
        self.average_train_loss_dict = {}
        self.average_train_acc_dict = {}
        # global model performance
        self.gfl_test_loss_dict = {}
        self.gfl_test_acc_dict = {}
        # local model performance (averaged across all clients)
        self.average_pfl_test_loss_dict = {}
        self.average_pfl_test_acc_dict = {}
        self.active_clients_indicies = None
        self.rounds = 0
#         # create a fake client on the server side; use for testing the performance of the global model
#         # trainset is only used for creating the label distribution
        self.server_side_client = kwargs['client_cstr'](
            kwargs['server_side_criterion'],
            kwargs['global_trainset'],
            kwargs['global_testset'],
            kwargs['server_side_client_config'],
            -1,
            device=fake_client_pyu,
            **kwargs)

    def select_clients(self, ratio):
        assert ratio > 0.0, "Invalid ratio. Possibly the server_config['participate_ratio'] is wrong."
        num_clients = int(ratio * self.num_clients)
        selected_indices = np.random.choice(range(self.num_clients), num_clients, replace=False)
        return selected_indices

    def testing(self, round, active_only, **kwargs):
        raise NotImplementedError

    def collect_stats(self, stage, round, active_only, **kwargs):
        raise NotImplementedError()

    def aggregate(self, client_uploads, round):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def save(self, filename, keep_clients_model=False):
        if not keep_clients_model:
            for client in self.clients_dict.values():
                client.model = None
                client.trainloader = None
                client.trainset = None
                client.new_state_dict = None
        self.server_side_client.trainloader = None
        self.server_side_client.trainset = None
        self.server_side_client.testloader = None
        self.server_side_client.testset = None
        save_to_pkl(self, filename)

    def summary_setup(self):
        info = "=" * 30 + "Run Summary" + "=" * 30
        info += "\nDataset:\n"
        info += f" dataset:{self.server_config['dataset']} | num_classes:{self.server_config['num_classes']}"
        partition = self.server_config['partition']
        info += f" | partition:{self.server_config['partition']}"
        if partition == 'iid-equal-size':
            info += "\n"
        elif partition in ['iid-diff-size', 'noniid-label-distribution']:
            info += f" | beta:{self.server_config['beta']}\n"
        elif partition == 'noniid-label-quantity':
            info += f" | num_classes_per_client:{self.server_config['num_classes_per_client']}\n "
        else:
            if 'shards' in partition.split('-'):
                pass
            else:
                raise ValueError(f" Invalid dataset partition strategy:{partition}!")
        info += "Server Info:\n"
        info += f" strategy:{self.server_config['strategy']} | num_clients:{self.server_config['num_clients']} | num_rounds: {self.server_config['num_rounds']}"
        info += f" | participate_ratio:{self.server_config['participate_ratio']} | drop_ratio:{self.server_config['drop_ratio']}\n"
        info += f"Clients Info:\n"
#         print('clients_dict',clients_dict[0])
#         client_config = self.clients_dict[0].client_config
#         client_config = client_config
        info += f" model:{self.client_config['model']} | num_epochs:{self.client_config['num_epochs']} | batch_size:{self.client_config['batch_size']}"
        info += f" | optimizer:{self.client_config['optimizer']} | inint lr:{self.client_config['client_lr']} | lr scheduler:{self.client_config['client_lr_scheduler']} | momentum: {self.client_config['sgd_momentum']} | weight decay: {self.client_config['sgd_weight_decay']}"
        print(info)
#         mdict = self.server_side_client.get_params()
# #         mdict_values = self.server_side_client.get_params_values()
#         print('mdict',mdict)
#         print(f" {client_config['model']}: size:{calculate_model_size(mdict):.3f} MB | num params:{sum(mdict[key].nelement() for key in mdict) / 1e6: .3f} M")

    def summary_result(self):
        raise NotImplementedError