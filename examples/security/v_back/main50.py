import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import os

from resnet50 import ResNet50
from resnet34 import ResNet34

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cpu")
cifardata = torch.load("./data/cifar102.pt")


def buildpos(x, lable):
    x = x.to(device1)
    lable = lable.to(device1)
    xlen = len(x)
    x1 = torch.zeros(xlen, 3, 32, 32)
    for i in range(xlen):
        a = random.randint(0, 4999)
        lab = lable[i].numpy()
        x1[i] = cifardata[lab, a]
    return x1


def buildneg(x, lable):
    x = x.to(device1)
    lable = lable.to(device1)
    xlen = len(x)
    x2 = torch.zeros(xlen, 3, 32, 32)
    lable2 = torch.zeros(xlen)
    for i in range(xlen):
        a = random.randint(0, 4999)
        lab1 = lable[i].numpy()
        lab = random.randint(0, 9)
        while lab == lab1:
            lab = random.randint(0, 9)
        x2[i] = cifardata[lab, a]
        lable2[i] = lab
    return x2, lable2


def main():
    # Check GPU, connect to it if it is available
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. GPU will be used for training.")
    else:
        device = "cpu"

    BEST_ACCURACY = 0
    batchsize = 128

    # Preparing Data
    print("==> Prepairing data ...")
    # Transformation on train data
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # transformation on validation data
    transform_validation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Download Train and Validation data and apply transformation
    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    validation_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_validation
    )

    # Put data into trainloader, specify batch_size
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize, shuffle=True, num_workers=2
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=100, shuffle=True, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Function to show CIFAR images
    def show_data(image):
        plt.imshow(np.transpose(image[0], (1, 2, 0)), interpolation="bicubic")
        plt.show()

    # show_data(train_data[0])

    # Need to import model a model
    model = ResNet50()
    # model = ResNet34()
    # model = CNN_batch()
    # Pass model to GPU

    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    criterion = nn.TripletMarginLoss(margin=30, p=2, reduce=True)
    length_train = len(train_data)
    length_validation = len(validation_data)
    print("Lentrain: ", length_train)
    print("Lenval: ", length_validation)
    print("Lentrainloader: ", len(train_loader))
    num_classes = 10

    # Training
    def train(epochs):
        global BEST_ACCURACY

        BEST_ACCURACY = 0
        dict = {
            "Train Loss": [],
            "Train Acc": [],
            "Validation Loss": [],
            "Validation Acc": [],
        }
        for epoch in range(epochs):
            print("\nEpoch:", epoch + 1, "/", epochs)
            cost = 0
            correct = 0
            total = 0
            woha = 0

            for i, (x, y) in enumerate(train_loader):
                woha += 1
                model.train()
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                yhat = model(x)
                yhat = yhat.reshape(-1, 10)

                posi = buildpos(x, y).to(device)
                nege, labneg = buildneg(x, y)
                nege = nege.to(device)
                mposi = model(posi).reshape(-1, 10)
                mnege = model(nege).reshape(-1, 10)

                loss = criterion(yhat, mposi, mnege)
                print(i, ":", loss)
                loss.backward()
                optimizer.step()
                cost += loss.item()

                _, yhat2 = torch.max(yhat.data, 1)
                correct += (yhat2 == y).sum().item()
                total += y.size(0)

            my_loss = cost / len(train_loader)
            my_accuracy = 100 * correct / length_train

            dict["Train Loss"].append(my_loss)
            dict["Train Acc"].append(my_accuracy)

            print("Tain Loss:", my_loss)
            print("Train Accuracy:", my_accuracy, "%")

            cost = 0
            correct = 0

            with torch.no_grad():
                for x, y in validation_loader:
                    x, y = x.to(device), y.to(device)
                    model.eval()
                    yhat = model(x)
                    yhat = yhat.reshape(-1, 10)
                    loss = crit(yhat, y)
                    cost += loss.item()

                    _, yhat2 = torch.max(yhat.data, 1)
                    correct += (yhat2 == y).sum().item()

            my_loss = cost / len(validation_loader)
            my_accuracy = 100 * correct / length_validation

            dict["Validation Loss"].append(my_loss)
            dict["Validation Acc"].append(my_accuracy)

            print("Validation Loss:", my_loss)
            print("Validation Accuracy:", my_accuracy, "%")

            # Save the model if you get best accuracy on validation data
            if my_accuracy > BEST_ACCURACY:
                BEST_ACCURACY = my_accuracy
                print("Saving the model ...")
                model.eval()
                if not os.path.isdir("model"):
                    os.mkdir("model")
                torch.save(model, "./model/resnet0050_trip_30.pt")
        # torch.save(model, './model/resnet50.pt')
        print("TRAINING IS FINISHED !!!")
        return dict

    results = train(200)

    print(results)

    plt.figure(1)
    plt.plot(results["Train Loss"], "b", label="training loss")
    plt.plot(results["Validation Loss"], "r", label="validation loss")
    plt.title("LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["training set", "validation set"], loc="center right")
    plt.savefig("Loss_ResNet50.png", dpi=300, bbox_inches="tight")

    plt.figure(2)
    plt.plot(results["Train Acc"], "b", label="training accuracy")
    plt.plot(results["Validation Acc"], "r", label="validation accuracy")
    plt.title("ACCURACY")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["training set", "validation set"], loc="center right")
    plt.savefig("Accuracy_ResNet50.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    """
    axs[0].plot(results['Train Loss'], 'b', label = 'training loss')
    axs[0].plot(results['Validation Loss'], 'r', label = 'validation loss')
    axs[0].set_title("LOSS")
    axs[0].set(xlabel="Epochs", ylabel="Loss")

    axs[1].plot(results['Train Acc'], 'b', label = 'training accuracy')
    axs[1].plot(results['Validation Acc'], 'r', label = 'validation accuracy')
    axs[1].set_title("ACCURACY")
    axs[1].set(xlabel="Epochs", ylabel="Accuracy")

    fig.tight_layout()
    plt.legend()
    plt.show()
    """


if __name__ == "__main__":
    main()
