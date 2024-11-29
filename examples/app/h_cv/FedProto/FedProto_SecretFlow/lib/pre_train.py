import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference, test_inference_new, LocalTest, test_inference_new_het, test_inference_new_het_cifar, test_inference_new_cifar

from models.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFemnist, Lenet
from utils import average_weights, exp_details, proto_aggregation, average_weights_het, agg_func, average_weights_per, average_weights_sem
import random
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
# from models import CNNFemnist
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])


def train(args, num_epoch, trainloader, testloader):
    model = Lenet(args=args)
    model.to(args.device)
    loss = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(self.parameters(),lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    criterion = nn.NLLLoss().to(args.device)
    for epoch in range(num_epoch):  # loop over the dataset multiple times
    # for epoch in range(1):
        timestart = time.time()

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            model.zero_grad()
            log_probs, protos = model(inputs)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print("i ",i)
            # if i % 500 == 499:  # print every 500 mini-batches
            if i % 100 == 0:
                print('[%d, %5d] loss: %.4f' %
                      (epoch, i, running_loss / 500))
                running_loss = 0.0
                _, predicted = torch.max(log_probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the %d tran images: %.3f %%' % (total,
                                                                                  100.0 * correct / total))
                total = 0
                correct = 0

        print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))

    print('Finished Training')

    print('Start Testing')
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            log_probs, protos = model(images)
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100.0 * correct / total))

    path = '../save/weights_'+str(num_epoch)+'ep.tar'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)

def test(args, num_epoch, testloader):
    print('Start Testing')
    # model = Lenet(args=args)
    # model.to(args.device)

    resnet18 = models.resnet18(pretrained=False, num_classes=args.num_classes)
    local_model = resnet18
    initial_weight = model_zoo.load_url(model_urls['resnet18'])
    initial_weight_1 = local_model.state_dict()
    for key in initial_weight.keys():
        if key[0:3] == 'fc.':
            initial_weight[key] = initial_weight_1[key]

    local_model.load_state_dict(initial_weight)
    local_model = local_model.to(args.device)
    # path = '../save/weights_'+str(num_epoch)+'ep.tar'
    #
    # checkpoint = torch.load(path, map_location=torch.device(args.device))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            log_probs, protos = local_model(images)
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100.0 * correct / total))

if __name__ == '__main__':
    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = '../data/cifar10/'
    num_epoch = 1

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=trans_cifar10_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=trans_cifar10_val)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=2)

    # train(args, num_epoch, trainloader, testloader)
    test(args, num_epoch, testloader)