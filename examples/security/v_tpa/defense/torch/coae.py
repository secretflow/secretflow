#!/usr/bin/env python
# coding=utf-8
import sys

sys.path.append("..")

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
import pdb

eps = 1.0e-35


def sharpen(probs, T):
    if len(probs.shape) == 1:
        temp = torch.pow(probs, 1.0 / T)
        temp = temp / (torch.pow(1.0 - temp, 1.0 / T) + temp)
    else:
        temp = torch.pow(probs, 1.0 / T)
        temp_sum = torch.sum(temp, dim=-1, keepdim=True)
        temp = temp / temp_sum
    return temp


class CoAE_Loss(_Loss):
    def __init__(self, ew=0.1, pw=10.0, nw=1.0):
        super().__init__()

        self.ew = ew
        self.pw = pw
        self.nw = nw

    def entropy(self, y_latent):
        result = torch.sum(-y_latent * torch.log(y_latent + eps), dim=1)
        # result = -y_latent * torch.log(y_latent + eps)
        return torch.mean(result)

    def cross_entropy(self, y_true, y_hat):
        # result = torch.sum(-y_true * torch.log(y_hat + eps), dim=1)
        result = torch.sum(-y_true * F.log_softmax(y_hat, dim=1), dim=1)
        return torch.mean(result)

    def forward(self, y_true, y_latent, y_hat):
        eloss = self.entropy(y_latent)
        ploss = self.cross_entropy(y_true, y_hat)
        nloss = self.cross_entropy(y_true, y_latent)

        loss = self.pw * ploss - self.ew * eloss - self.nw * nloss
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, T=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.T = T

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, y_true):
        y_latent = self.encoder(y_true)
        y_latent = sharpen(y_latent, self.T)

        y_hat = self.decoder(y_latent)
        return y_latent, y_hat


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 10
    ew = 0.00
    loss_fn = CoAE_Loss(ew=ew)
    model = AutoEncoder(num_classes, (6 * num_classes + 2) ** 2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # epochs = 100
    epochs = 50
    batch_size = 128
    train_size, test_size = 30000, 10000
    T = 0.025

    y_rand = torch.rand(train_size, num_classes)
    y_train = sharpen(F.softmax(y_rand, dim=1), T=T)

    y_rand = torch.rand(test_size, num_classes)
    y_test = sharpen(F.softmax(y_rand, dim=1), T=T).to(device)
    y_labels_test = torch.argmax(y_test, dim=1).to(device)

    y_onehot = (
        F.one_hot(
            torch.from_numpy(np.random.randint(num_classes, size=(test_size,))),
            num_classes=num_classes,
        )
        .float()
        .to(device)
    )
    y_labels_onehot = torch.argmax(y_onehot, dim=1).to(device)

    def train_on_batch(y_true):
        y_latent, y_hat = model(y_true)
        loss = loss_fn(y_true, y_latent, y_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, y_latent, y_hat

    dataset = TensorDataset(y_train)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    for i in range(epochs):
        total_loss, batch_count = 0, 0
        for y_true in dataloader:
            y_true = y_true[0].to(device)
            loss, y_latent, y_hat = train_on_batch(y_true)
            total_loss = total_loss + loss
            batch_count += 1

        if (i + 1) % 10 == 0:
            with torch.no_grad():
                y_latent, y_hat = model(y_test)
                y_latent_onehot, y_hat_onehot = model(y_onehot)
                acc_p = torch.sum(torch.argmax(y_hat, dim=1) == y_labels_test) / len(
                    y_labels_test
                )
                acc_n = torch.sum(torch.argmax(y_latent, dim=1) != y_labels_test) / len(
                    y_labels_test
                )
                acc_p_onehot = torch.sum(
                    torch.argmax(y_hat_onehot, dim=1) == y_labels_onehot
                ) / len(y_labels_onehot)
                acc_n_onehot = torch.sum(
                    torch.argmax(y_latent_onehot, dim=1) != y_labels_onehot
                ) / len(y_labels_onehot)
                cur_loss = total_loss / batch_count
            print(
                "epoch",
                i + 1,
                "acc_p:",
                acc_p.item(),
                "acc_n:",
                acc_n.item(),
                "acc_p_onehot:",
                acc_p_onehot.item(),
                "acc_n_onehot:",
                acc_n_onehot.item(),
                "loss:",
                cur_loss.item(),
            )

    model_path = "../../trained_model/{}-{:.2f}.pt".format(num_classes, ew)
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    print("end")
