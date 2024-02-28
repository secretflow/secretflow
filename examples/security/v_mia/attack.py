# Copyright 2023 Zeping Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn
import torch.optim as optim
from piq import SSIMLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import *
from torchvision.transforms.functional import *
from torchvision.transforms.transforms import *
from torchvision.utils import save_image
from tqdm import tqdm

if __name__ == "__main__":
    batch_size = 128
    train_epoches = 100
    log_epoch = 4
    class_num = 10
    root_dir = "./log/splitlearning/logZZPMAIN.attack"
    feature_pkl = "/path/to/your/client.pkl"
    cls_pkl = "/path/to/your/server.pkl"
    log_dir = root_dir
    model_dir = root_dir
    h = 32
    w = 32
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=log_dir)

    data_workers = 4

    transform = Compose(
        [
            Resize((h, w)),
            ToTensor(),
        ]
    )

    mnist_train_ds = MNIST(
        root="./datasets", train=True, transform=transform, download=True
    )
    mnist_test_ds = MNIST(
        root="./datasets", train=False, transform=transform, download=True
    )

    mnist_train_ds_len = len(mnist_train_ds)
    mnist_test_ds_len = len(mnist_test_ds)

    train_ds = mnist_train_ds
    test_ds = mnist_test_ds

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
    )

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    class FeatureExtracter(nn.Module):
        def __init__(self):
            super(FeatureExtracter, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
            self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(512)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()

        def forward(self, x: Tensor):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            x = self.relu3(x)
            x = x.view(-1, 8192)
            return x

    class CLS(nn.Module):
        def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
            super(CLS, self).__init__()
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)

        def forward(self, x):
            feature = self.bottleneck(x)
            out = self.fc(feature)
            return [out, feature]

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 512, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(128, 1, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.sigmod = nn.Sigmoid()

        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.sigmod(x)
            return x

    feature_extractor = FeatureExtracter().train(False).to(output_device)
    cls = CLS(8192, class_num, 50).train(False).to(output_device)
    myinversion = Inversion(50).train(True).to(output_device)

    optimizer = optim.Adam(
        myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    assert os.path.exists(feature_pkl)
    feature_extractor.load_state_dict(
        torch.load(open(feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(cls_pkl)
    cls.load_state_dict(torch.load(open(cls_pkl, "rb"), map_location=output_device))

    feature_extractor.requires_grad_(False)
    cls.requires_grad_(False)

    os.makedirs(f"{log_dir}/input/", exist_ok=True)
    os.makedirs(f"{log_dir}/output/", exist_ok=True)

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, (im, label) in enumerate(tqdm(test_dl, desc=f"epoch {epoch_id} ")):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            feature8192 = feature_extractor.forward(im)
            out, feature = cls.forward(feature8192)
            rim = myinversion.forward(feature)
            ssim = SSIMLoss()(rim, im)
            loss = ssim
            loss.backward()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            writer.add_scalar("loss", loss, epoch_id)
            writer.add_scalar("ssim", ssim, epoch_id)
            save_image(im.detach(), f"{log_dir}/input/{epoch_id}.png")
            save_image(rim.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(
                os.path.join(model_dir, f"myinversion_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(myinversion.state_dict(), f)

        if epoch_id % log_epoch == 0:
            with torch.no_grad():
                myinversion.eval()
                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(tqdm(train_dl, desc=f"priv")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    out, feature = cls.forward(feature8192)
                    rim = myinversion.forward(feature)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                writer.add_scalar("priv ssim", ssimlossavg, epoch_id)

                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(tqdm(test_dl, desc=f"aux")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    out, feature = cls.forward(feature8192)
                    rim = myinversion.forward(feature)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                writer.add_scalar("aux ssim", ssimlossavg, epoch_id)
    writer.close()
