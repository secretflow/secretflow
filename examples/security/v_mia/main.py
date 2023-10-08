import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy
from torchplus.distributed import FederatedAverage

if __name__ == '__main__':
    batch_size = 128
    train_epoches = 48
    log_epoch = 4
    class_num = 10
    client_num = 2
    root_dir = "D:/log/splitlearning/logZZPMAIN"
    h = 32
    w = 32

    init = Init(seed=9970, log_root_dir=root_dir,
                backup_filename=__file__, tensorboard=True, comment=f'main split learning MNIST')
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([
        Resize((h, w)),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    priv_ds = MNIST(root='./data', train=True,
                    transform=transform, download=True)
    aux_ds = MNIST(root='./data', train=False,
                   transform=transform, download=True)

    priv_ds_len = len(priv_ds)
    aux_ds_len = len(aux_ds)

    priv_ds_0, priv_ds_1 = random_split(
        priv_ds, [priv_ds_len*1//client_num, priv_ds_len-priv_ds_len*1//client_num])
    aux_ds_0, aux_ds_1 = random_split(
        aux_ds, [aux_ds_len*1//client_num, aux_ds_len-aux_ds_len*1//client_num])

    train_dl = DataLoader(dataset=priv_ds, batch_size=batch_size,
                          shuffle=False, num_workers=data_workers, drop_last=False, pin_memory=True)

    test_dl = DataLoader(dataset=aux_ds, batch_size=batch_size,
                         shuffle=False, num_workers=data_workers, drop_last=False, pin_memory=True)

    train_dl_0 = DataLoader(dataset=priv_ds_0, batch_size=batch_size,
                            shuffle=True, num_workers=data_workers, drop_last=True, pin_memory=True)

    train_dl_1 = DataLoader(dataset=priv_ds_1, batch_size=batch_size,
                            shuffle=True, num_workers=data_workers, drop_last=True, pin_memory=True)

    test_dl_0 = DataLoader(dataset=aux_ds_0, batch_size=batch_size,
                           shuffle=False, num_workers=data_workers, drop_last=False, pin_memory=True)

    test_dl_1 = DataLoader(dataset=aux_ds_1, batch_size=batch_size,
                           shuffle=False, num_workers=data_workers, drop_last=False, pin_memory=True)

    train_dl_list = [train_dl_0, train_dl_1]
    test_dl_list = [test_dl_0, test_dl_1]

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
            x = self.bottleneck(x)
            x = self.fc(x)
            return x

    client_0 = FeatureExtracter().train(True).to(output_device)
    client_1 = FeatureExtracter().train(True).to(output_device)
    server = CLS(8192, class_num, 50).train(True).to(output_device)

    client_list = [client_0, client_1]

    optimizer_cli_0 = optim.Adam(client_0.parameters(), lr=0.0002,
                                 betas=(0.5, 0.999), amsgrad=True)
    optimizer_cli_1 = optim.Adam(client_1.parameters(), lr=0.0002,
                                 betas=(0.5, 0.999), amsgrad=True)
    optimizer_server = optim.Adam(server.parameters(), lr=0.0002,
                                  betas=(0.5, 0.999), amsgrad=True)

    optimizer_cli_list = [optimizer_cli_0, optimizer_cli_1]

    for epoch_id in tqdm(range(1, train_epoches+1), desc='Total Epoch'):
        for client_id in range(client_num):
            for i, (im, label) in enumerate(tqdm(train_dl_list[client_id], desc=f'Client {client_id} epoch {epoch_id}')):
                im = im.to(output_device)
                label = label.to(output_device)
                bs, c, h, w = im.shape
                optimizer_cli_list[client_id].zero_grad()
                optimizer_server.zero_grad()
                feature = client_list[client_id].forward(im)
                out = server.forward(feature)
                ce = nn.CrossEntropyLoss()(out, label)
                loss = ce
                loss.backward()
                optimizer_cli_list[client_id].step()
                optimizer_server.step()

        client_list = FederatedAverage(client_list)

        if epoch_id % log_epoch == 0:
            train_ca = ClassificationAccuracy(class_num)
            after_softmax = F.softmax(out, dim=-1)
            predict = torch.argmax(after_softmax, dim=-1)
            train_ca.accumulate(label=label, predict=predict)
            acc_train = train_ca.get()
            writer.add_scalar('loss', loss, epoch_id)
            writer.add_scalar('acc_training', acc_train, epoch_id)
            with open(os.path.join(log_dir, f'client_{epoch_id}.pkl'), 'wb') as f:
                torch.save(client_0.state_dict(), f)
            with open(os.path.join(log_dir, f'server_{epoch_id}.pkl'), 'wb') as f:
                torch.save(server.state_dict(), f)

            with torch.no_grad():
                client_0.eval()
                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(train_dl, desc='testing train')):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature = client_0.forward(im)
                    out = server.forward(feature)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss/r
                acc_test = test_ca.get()
                writer.add_scalar('train loss', celossavg, epoch_id)
                writer.add_scalar('acc_train', acc_test, epoch_id)

                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(test_dl, desc='testing test')):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature = client_0.forward(im)
                    out = server.forward(feature)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss/r
                acc_test = test_ca.get()
                writer.add_scalar('test loss', celossavg, epoch_id)
                writer.add_scalar('acc_test', acc_test, epoch_id)

    writer.close()
