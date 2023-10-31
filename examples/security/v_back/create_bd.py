import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.autograd import Variable


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
criterion = nn.CrossEntropyLoss()


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def createbd(x, gradient_step, cicle, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lab = torch.tensor([x]).to(device)
    print(lab.shape)
    tensor = torch.zeros(3, 32, 32).to(device)
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=True)

    for i in range(cicle):
        tensor.requires_grad = True

        yhat = model(tensor)
        yhat = yhat.reshape(-1, 10)
        # print(yhat.shape)

        loss = criterion(yhat, lab)
        loss.backward(retain_graph=True)
        # print(tensor.grad)
        print(loss)

        with torch.no_grad():
            # apply gradient descent
            tensor[:, :, 23:30, 23:30] = (
                tensor[:, :, 23:30, 23:30]
                - gradient_step * tensor.grad[:, :, 23:30, 23:30]
            )

    ""
    for i in range(350):
        tensor.requires_grad = True

        yhat = model(tensor)
        yhat = yhat.reshape(-1, 10)
        # print(yhat.shape)

        loss = criterion(yhat, lab)
        loss.backward(retain_graph=True)
        # print(tensor.grad)
        print(loss)

        with torch.no_grad():
            # apply gradient descent
            tensor[:, :, 23:30, 23:30] = (
                tensor[:, :, 23:30, 23:30] - 0.001 * tensor.grad[:, :, 23:30, 23:30]
            )
    ""
    for i in range(300):
        tensor.requires_grad = True

        yhat = model(tensor)
        yhat = yhat.reshape(-1, 10)
        # print(yhat.shape)

        loss = criterion(yhat, lab)
        loss.backward(retain_graph=True)
        # print(tensor.grad)
        print(loss)

        with torch.no_grad():
            # apply gradient descent
            tensor[:, :, 23:30, 23:30] = (
                tensor[:, :, 23:30, 23:30] - 0.0001 * tensor.grad[:, :, 23:30, 23:30]
            )
    ""
    for i in range(300):
        tensor.requires_grad = True

        yhat = model(tensor)
        yhat = yhat.reshape(-1, 10)
        # print(yhat.shape)

        loss = criterion(yhat, lab)
        loss.backward(retain_graph=True)
        # print(tensor.grad)
        print(loss)

        with torch.no_grad():
            # apply gradient descent
            tensor[:, :, 23:30, 23:30] = (
                tensor[:, :, 23:30, 23:30] - 0.00005 * tensor.grad[:, :, 23:30, 23:30]
            )
    ""
    torch.save(tensor, "./backdoor/bdlab{}.pt".format(x))
    print(model(tensor).reshape(-1, 10))
    print(lab)
    image = tensor.squeeze(0).detach().clone().to("cpu")
    print(image.shape)
    # imshow(image)


# 生成后门触发器
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)
    model = torch.load("model/resnet0050.pt")
    model.eval()

    gradient_step = 0.01

    createbd(9, gradient_step, 80, model)


if __name__ == "__main__":
    main()
