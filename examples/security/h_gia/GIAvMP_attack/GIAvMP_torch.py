# Copyright 2024 Ant Group Co., Ltd.
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
import copy
import os

import torch
from torchvision.utils import save_image

from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.security.privacy.mechanism.mechanism_fl import GaussianModelDP
from secretflow_fl.security.privacy.strategy_fl import DPStrategyFL

from .utils import DLGinverse, train_malicious_params


# AttackCallback of class GIAvMP(AttackCallback):
class GIAvMP(AttackCallback):
    """
    Implemention of GIAvMP attack, a gradient inversion attack via model poisoning. GIAvMP assumes that the central server is "malicious and active", who will send malicious gradients to poison the model of the victim party. The model poisoning can mitigate the confusion of aggregated gradients significantly. Then the attacker can reconstruct the raw data of the victim party even in large-batch FL scenarios.

    Attributes:
        attack_party: attack party.
        victim_party: victim party.
        attack_configs: detailed configurations of SME attack.

    """

    def __init__(
        self, attack_party, victim_party, aux_dataset, attack_configs, **kwargs
    ):
        super().__init__(**kwargs)

        self.attack_party = attack_party
        self.victim_party = victim_party
        self.attack_configs = attack_configs
        self.aux_dataset = aux_dataset
        self.attacker = None

    def on_train_begin(self, logs=None):

        # get the current global model params, we directly get params of the victim model for convenience
        # self.global_params = (
        #     self._workers[self.victim_party].get_weights().to(self.attack_party)
        # )

        # init the attacker
        def init_attacker(attack_configs):
            return GIAvMP_attacker(attack_configs)

        self.attacker = self.attack_party(init_attacker)(self.attack_configs)

        # construct malicious parameters
        def construct_malicious_params(attacker, aux_dataset):
            global_params = attacker.construct_malicious_params(aux_dataset)
            return global_params

        global_params = self.attack_party(construct_malicious_params)(
            self.attacker, self.aux_dataset
        )

        # the victim party receive the malicious parameters
        self._workers[self.victim_party].set_weights(
            global_params.to(self.victim_party)
        )
        return

    def on_train_end(self, logs=None):

        # get the victim party's model parameters
        victim_params = (
            self._workers[self.victim_party].get_weights().to(self.attack_party)
        )

        # the raw data of victim is only used to evaluate the attack performance
        # the attacker has no access to the raw data in practical scenario
        def get_victim_trainloader(victim_worker: BaseTorchModel):
            return victim_worker.train_set

        victim_trainloader = (
            self._workers[self.victim_party]
            .apply(get_victim_trainloader)
            .to(self.attack_party)
        )

        def GIAvMP_attack(victim_params, victim_trainloader, attacker):
            attacker.victim_params = victim_params
            attacker.victim_trainloader = victim_trainloader

            attacker.reconstruct_raw_data()

            return

        self.attack_party(GIAvMP_attack)(
            victim_params, victim_trainloader, self.attacker
        )


# define the class GIAvMP_attacker
class GIAvMP_attacker:
    def __init__(self, attack_configs):
        self.global_params_before = None
        self.victim_params = None
        self.victim_trainloader = None
        self.attack_configs = attack_configs

        self.raw_images = []

    def construct_malicious_params(self, aux_dataset=None):

        # init the global params
        self.global_net = self.attack_configs['model']()
        global_params = [
            v.detach().clone().numpy() for v in list(self.global_net.parameters())
        ]

        # load malicious params
        if not self.attack_configs['trainMP']:

            if self.attack_configs['model'].__name__ == 'FCNNmodel':

                fc_mp = torch.load(
                    self.attack_configs['path_to_malicious_params'],
                    map_location=torch.device('cpu'),
                )

                # insert the malicious params into the global params
                global_params[0] = fc_mp['linear0.weight'].numpy()
                global_params[1] = fc_mp['linear0.bias'].numpy()

        # train malicious params
        else:
            global_params = train_malicious_params(
                global_net=self.global_net,
                aux_dataset=aux_dataset,
                global_params=global_params,
                attack_configs=self.attack_configs,
            )

        self.global_params_before = copy.deepcopy(global_params)
        return global_params

    # reconstruct raw data
    def reconstruct_raw_data(self):
        # create the directory to save the recovered images
        os.makedirs(self.attack_configs['path_to_res'], exist_ok=True)

        # get the raw images for validating the attack performance
        raw_images = []
        raw_labels = []
        for img, l in self.victim_trainloader:
            raw_images.append(img)
            raw_labels.append(l)
        self.raw_images = torch.cat(raw_images)
        self.raw_labels = torch.cat(raw_labels)

        # compute the gradient
        grad = [g - v for g, v in zip(self.global_params_before, self.victim_params)]

        # test DP defense
        # dp_strategy = DPStrategyFL(
        #     model_gdp=GaussianModelDP(
        #         noise_multiplier=0.001,
        #         num_clients=1,
        #         l2_norm_clip=1.0,
        #     )
        # )
        # grad = dp_strategy.model_gdp(grad)

        # GIAvMP for FCNN model
        if self.attack_configs['model'].__name__ == 'FCNNmodel':
            # recover images from the neurons of the 1st FC layer with analytic method
            recovered_images = []
            for i in range(len(grad[1])):
                if grad[1][i] == 0:
                    data = torch.zeros(grad[0][i].shape)
                else:
                    data = grad[0][i] / grad[1][i]
                data = data.reshape(self.attack_configs['data_size'])
                recovered_images.append(torch.Tensor(data))

            recovered_images = torch.stack(recovered_images)
            recovered_images = recovered_images.float()

            # save the recovered data from all neurons in the 1st FC layer
            save_image(
                tensor=recovered_images.cpu().detach(),
                fp=os.path.join(
                    self.attack_configs['path_to_res'], 'recovered_images.png'
                ),
            )

            # find the most similar image in recovered_images for each raw image
            concatenated_images = []
            psnrs = 0
            mses = 0
            for raw_image in self.raw_images:
                similarities = [
                    torch.nn.functional.cosine_similarity(
                        raw_image.view(-1), recovered_image.view(-1), dim=0
                    )
                    for recovered_image in recovered_images
                ]
                most_similar_idx = torch.argmax(torch.tensor(similarities))
                most_similar_image = recovered_images[most_similar_idx]
                concatenated_images.append(
                    torch.cat((raw_image, most_similar_image), dim=2)
                )
                mse = torch.nn.functional.mse_loss(raw_image, most_similar_image)
                mses += mse
                psnr = 10 * torch.log10(1 / mse)
                psnrs += psnr

            concatenated_images = torch.stack(concatenated_images)
            avg_psnr = psnrs / len(self.raw_images)
            avg_mse = mses / len(self.raw_images)
            print("avg psnr {}".format(avg_psnr))
            print("avg mse {}".format(avg_mse))

            # save the concatenated images
            save_image(
                tensor=concatenated_images.cpu().detach(),
                fp=os.path.join(
                    self.attack_configs['path_to_res'], 'raw_and_recovered_images.png'
                ),
            )

        # GIAvMP for CNN model
        elif self.attack_configs['model'].__name__ == 'CNNmodel':
            # recover features from the neurons of the 1st FC layer with analytic method
            recovered_features = []
            for i in range(len(grad[6])):
                if grad[7][i] == 0:
                    data = torch.zeros(grad[6][i].shape)
                else:
                    data = grad[6][i] / grad[7][i]
                recovered_features.append(torch.Tensor(data))

            recovered_features = torch.stack(recovered_features)
            # recovered_features = recovered_features.float()

            raw_features = self.global_net.body(self.raw_images)
            raw_features = raw_features.reshape(raw_features.size(0), -1)

            # find the most similar features in recovered_features for each feature in raw_features using cosine similarity
            selected_features = []
            for feature in raw_features:
                similarities = [
                    torch.nn.functional.cosine_similarity(feature, rf, dim=0)
                    for rf in recovered_features
                ]
                most_similar_idx = torch.argmax(torch.tensor(similarities))
                most_similar_feature = recovered_features[most_similar_idx]
                selected_features.append(most_similar_feature)

            # inverse raw images with DLG method
            DLGinverse(
                net=self.global_net,
                gt_data=self.raw_images,
                gt_label=self.raw_labels,
                gt_embed=recovered_features,
                attack_configs=self.attack_configs,
                original_dy_dx=grad,
            )
