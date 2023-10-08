import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, save_excel,save_image2
from torchplus.nn import PixelLoss

if __name__ == '__main__':
    batch_size = 32
    class_num = 10
    root_dir = "D:/log/splitlearning/logZZPMAIN.test"
    feature_pkl = "D:/log/splitlearning/logZZPMAIN/Mar19_19-43-15_zzp-asus_main split learning MNIST/client_48.pkl"
    cls_pkl = "D:/log/splitlearning/logZZPMAIN/Mar19_19-43-15_zzp-asus_main split learning MNIST/server_48.pkl"
    inv_pkl = "D:/log/splitlearning/logZZPMAIN.attack/Model_Apr23_15-44-47_zzp-asus_mnist attack 50 sfl/myinversion_100.pkl"
    h = 32
    w = 32

    init = Init(seed=9970, log_root_dir=root_dir,
                backup_filename=__file__, tensorboard=True, comment=f'MNIST export')
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 4

    transform = Compose([
        Resize((h, w)),
        ToTensor()
    ])

    mnist_train_ds = MNIST(root='E:/datasets', train=True,
                             transform=transform, download=True)
    mnist_test_ds = MNIST(root='E:/datasets', train=False,
                            transform=transform, download=True)

    mnist_train_ds_len = len(mnist_train_ds)
    mnist_test_ds_len = len(mnist_test_ds)

    train_ds = mnist_train_ds
    test_ds = mnist_test_ds

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)


    # for evaluate
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size,
                          shuffle=False, num_workers=data_workers, drop_last=False)
    # for attack
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=data_workers, drop_last=False)

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
    myinversion = Inversion(50).train(False).to(output_device)

    assert os.path.exists(feature_pkl)
    feature_extractor.load_state_dict(torch.load(
        open(feature_pkl, 'rb'), map_location=output_device))

    assert os.path.exists(cls_pkl)
    cls.load_state_dict(torch.load(
        open(cls_pkl, 'rb'), map_location=output_device))

    assert os.path.exists(inv_pkl)
    myinversion.load_state_dict(torch.load(
        open(inv_pkl, 'rb'), map_location=output_device))
    
    os.makedirs(f'{log_dir}/priv/input/', exist_ok=True)
    os.makedirs(f'{log_dir}/priv/output/', exist_ok=True)
    os.makedirs(f'{log_dir}/aux/input/', exist_ok=True)
    os.makedirs(f'{log_dir}/aux/output/', exist_ok=True)

    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f'priv')):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            feature8192 = feature_extractor.forward(im)
            out, feature = cls.forward(feature8192)
            rim = myinversion.forward(feature)
            save_image2(im.detach(), f'{log_dir}/priv/input/{i}.png', nrow=4)
            save_image2(rim.detach(), f'{log_dir}/priv/output/{i}.png', nrow=4)

        for i, (im, label) in enumerate(tqdm(test_dl, desc=f'aux')):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            feature8192 = feature_extractor.forward(im)
            out, feature = cls.forward(feature8192)
            rim = myinversion.forward(feature)
            save_image2(im.detach(), f'{log_dir}/aux/input/{i}.png', nrow=4)
            save_image2(rim.detach(), f'{log_dir}/aux/output/{i}.png', nrow=4)

    writer.close()
