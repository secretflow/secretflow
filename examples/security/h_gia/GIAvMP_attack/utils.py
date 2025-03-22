# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


def train_malicious_params(
    global_net,
    aux_dataset,
    global_params,
    attack_configs,
    lr=0.001,
    epochs=5,
    K=4,
):
    """Trains malicious parameters for the GIAvMP attack.

    Args:
        global_net: The global neural network model.
        aux_dataset: Auxiliary dataset used for training malicious parameters.
        global_params: the global parameters.
        attack_configs: Attack configurations.
        lr: Learning rate for training.
        epochs: Number of training epochs.
        K: Number of neurons to select.
    """
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    # load the dataset
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            return sample

    dataset = CustomDataset(aux_dataset)
    trainLoader = DataLoader(dataset, batch_size=64, shuffle=True)

    if attack_configs['model'].__name__ == 'FCNNmodel':
        width_out, width_in = global_params[0].shape
    elif attack_configs['model'].__name__ == 'CNNmodel':
        width_out, width_in = global_params[6].shape[0], global_params[6].shape[1]
        netemb = copy.deepcopy(global_net.body).to(device)

    netTR = torch.nn.Sequential(
        OrderedDict(
            [
                ('flatten', torch.nn.Flatten()),
                ('linear0', torch.nn.Linear(width_in, width_out)),
                ('sig', torch.nn.Sigmoid()),
            ]
        )
    )

    optimizer = torch.optim.Adam(netTR.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(epochs * 0.75)], 0.1
    )

    # init the malicious parameters of the first FC layer with Gaussian distribution
    RMPinit(2, 0.97, width_in, width_out, netTR)

    netTR.to(device)
    # train the malicious parameters
    for iter in range(epochs):
        running_loss = 0

        # the table used to count the neurons that have been selected as SDANs
        NeuralTable = torch.zeros(width_out).to(device)

        for i, data in enumerate(trainLoader):
            optimizer.zero_grad()

            img, label = data[0].to(device), data[1].to(device)

            # if the model is CNN, we use the features after the convolution layers to train malicious parameters
            if attack_configs['model'].__name__ == 'CNNmodel':
                netemb.eval()
                img = netemb(img)
                img = img.view(img.size(0), -1)

            output = netTR(img)

            loss = MPloss(output, K, NeuralTable, device)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        scheduler.step()

        running_loss /= len(trainLoader)
        print('epoch %d: loss  %.6f' % (iter + 1, running_loss))

    # insert the malicious parameters into the global parameters
    if attack_configs['model'].__name__ == 'FCNNmodel':
        global_params[0] = netTR.state_dict()['linear0.weight'].cpu().numpy()
        global_params[1] = netTR.state_dict()['linear0.bias'].cpu().numpy()
    elif attack_configs['model'].__name__ == 'CNNmodel':
        global_params[6] = netTR.state_dict()['linear0.weight'].cpu().numpy()
        global_params[7] = netTR.state_dict()['linear0.bias'].cpu().numpy()

    # save the malicious parameters
    torch.save(
        global_params,
        os.path.join(
            "./examples/security/h_gia/GIAvMP_attack/malicious_params",
            '{}-MP.pth'.format(attack_configs['model'].__name__, iter + 1),
        ),
    )

    return global_params


def DLGinverse(
    net,
    gt_data,
    gt_label,
    gt_embed,
    attack_configs,
    original_dy_dx,
    r=1000,
    lr=0.01,
    loss_factors=[1e-3, 1, 1e-2],
):
    """Inverses raw images using DLG method.

    Args:
        net: The neural network model.
        gt_data: Ground truth data.
        gt_label: Ground truth labels.
        gt_embed: Ground truth embeddings.
        attack_configs: Attack configurations.
        original_dy_dx: Original gradients.
        r: Number of iterations.
        lr: Learning rate.
        loss_factors: Loss factors for different loss components.
    """
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    b1, b2, b3 = loss_factors

    criterion = F.cross_entropy
    tv_loss = TVLoss()

    net = copy.deepcopy(net).to(device)
    net.eval()
    embd = net.body

    # generate dummy data and label
    dummy_data = torch.randn(*gt_data.size()).to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([dummy_data], lr=lr)

    scheduler = MultiStepLR(
        optimizer, milestones=[r // 2.667, r // 1.6, r // 1.142], gamma=0.1
    )

    original_dy_dx = [torch.from_numpy(g).to(device) for g in original_dy_dx]
    gt_embed = [g.to(device) for g in gt_embed]

    psnr_max = 0
    record_loss = []
    record_psnr = []
    for iters in range(r):
        optimizer.zero_grad()
        net.zero_grad()

        dummy_pred = net(dummy_data)
        dummy_loss = criterion(dummy_pred, gt_label.to(device))
        dummy_dy_dx = torch.autograd.grad(
            dummy_loss, net.parameters(), create_graph=True
        )
        # the gradient matching loss
        loss1 = calculate_cosine_distances(dummy_dy_dx, original_dy_dx)  # l2  sim

        dummy_f = embd(dummy_data)
        dummy_f = dummy_f.view(dummy_f.size(0), -1)
        # the feature matching loss
        loss2 = calculate_cosine_distances(dummy_f, gt_embed)  # l2  sim
        # the TV loss
        loss3 = tv_loss(dummy_data)

        loss = b1 * loss1 + b2 * loss2 + b3 * loss3

        loss.backward()

        optimizer.step()
        scheduler.step()

        dummy_data.data = torch.clamp(dummy_data, 0, 1)

        if (iters + 1) % 50 == 0:
            mse = torch.mean(
                (gt_data.detach() - dummy_data.detach().clone().cpu()) ** 2,
                dim=(1, 2, 3),
            )
            psnr = 10 * torch.log10(1 / mse)
            psnr_mean = torch.mean(psnr)
            print(
                'rounds: [%d|%d]  ||  lr: %.4f  ||  loss: %.6f  ||  PSNR: %.6f'
                % (iters + 1, r, scheduler.get_last_lr()[0], loss.item(), psnr_mean)
            )

            figs = []
            for i in range(len(gt_data)):
                figs.append(gt_data[i])

                similarities = [
                    torch.nn.functional.mse_loss(
                        gt_data[i].view(-1), recovered_image.cpu().view(-1)
                    )
                    for recovered_image in dummy_data
                ]
                most_similar_idx = torch.argmin(torch.tensor(similarities))
                most_similar_image = dummy_data[most_similar_idx]

                figs.append(most_similar_image.cpu())
            figs = torch.stack(figs).float()
            # if psnr_max <= psnr_mean.item():
            save_image(
                figs.clone().detach().cpu(),
                os.path.join(
                    attack_configs['path_to_res'],
                    'raw_and_recovered_images_CNN.png',
                ),
            )
            # save_image(dummy_data[:16].clone().detach().cpu(), filename+'rdataB16.png', nrow=4)
            record_loss.append(loss.item())
            record_psnr.append(psnr_mean.item())
            psnr_max = max(psnr_mean.item(), psnr_max)

    return psnr_max


# generate malicious parameters with Gaussian distribution for each neuron
def randomMP(f_dim, b, s):
    randomIndex = np.random.choice(f_dim, f_dim, replace=False)
    P = randomIndex[: f_dim // 2]
    N = randomIndex[f_dim // 2 :]
    zN = np.random.normal(loc=0, scale=b, size=f_dim // 2)
    zP = zN.copy() * s * (-1)
    np.random.shuffle(zN)
    np.random.shuffle(zP)
    rMP = np.zeros(f_dim)
    for i in range(f_dim // 2):
        rMP[N[i]] = zN[i]
        rMP[P[i]] = zP[i]
    return rMP


# generate malicious parameters with Gaussian distribution for one FC layer
def RMPinit(b, s, f_dim, L, net):
    weight = []
    for i in range(L):
        rmp = randomMP(f_dim, b, s)
        weight.append(rmp)
    weight = torch.Tensor(np.array(weight))
    bias = torch.zeros(L)
    p = net.state_dict()
    p['linear0.weight'] = weight
    p['linear0.bias'] = bias
    net.load_state_dict(p)


def get_maxindex(output, k, t, device):
    avgt = torch.mean(t)

    output = output.clone().detach()
    maxindex = torch.Tensor([]).long().to(device)
    for i in range(len(output)):
        outputi = output[i]
        outputi[maxindex] = -1
        outputi[t > avgt] = -1
        _, index = torch.topk(outputi, k)
        maxindex = torch.concat((maxindex, index))
    t[maxindex] += 1
    res = []
    for i in range(k):
        item = [maxindex[k * j + i] for j in range(len(output))]
        res.append(torch.Tensor(item).long().to(device))
    return res


# the training loss for malicious parameters
def MPloss(output, k, t, device):
    loss = 0

    lossFun = nn.CrossEntropyLoss()
    # choose neurons to be SDANs according to the outputs of the FC layer
    maxindex = get_maxindex(output, k, t, device)
    for i in range(k):
        loss += lossFun(output, maxindex[i])
    loss /= k

    return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def reconstruction_costs(gradients, input_gradient, cost_fn='l2'):

    indices = torch.arange(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0

        for i in indices:
            costs -= (trial_gradient[i] * input_gradient[i]).sum()
            pnorm[0] += trial_gradient[i].pow(2).sum()
            pnorm[1] += input_gradient[i].pow(2).sum()

        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def calculate_cosine_distances(tensor_list1, tensor_list2):
    distances = 0
    # 假设两个列表长度相同
    for tensor1, tensor2 in zip(tensor_list1, tensor_list2):
        # 计算余弦相似度
        similarity = (tensor1 * tensor2).sum()
        pnorm1 = tensor1.pow(2).sum()
        pnorm2 = tensor2.pow(2).sum()
        # 计算余弦距离
        distance = 1 - similarity / pnorm1.sqrt() / pnorm2.sqrt()
        distances += distance
    distances /= len(tensor_list1)
    return distances
