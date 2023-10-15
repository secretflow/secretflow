"""
    testing example for prune_gradient_strategy within pytorch task
"""
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import Accuracy, Precision
from torchvision import datasets, models, transforms

import secretflow as sf
from dgl_utils import (ConvNet, LeNet, ResNet, ResNet34,
                       cross_entropy_for_onehot, label_to_onehot)
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.utils import BaseModule, TorchModel
from secretflow.security.aggregation import (PlainAggregator, SecureAggregator,
                                             SecurePruneAggregator)
from secretflow.utils.simulation.datasets import load_mnist

sf.init(["alice", "bob", "charlie"], address="local")
alice, bob, charlie = sf.PYU("alice"), sf.PYU("bob"), sf.PYU("charlie")

(train_data, train_label), (test_data, test_label) = load_mnist(
    parts={alice: 0.5, bob: 0.5},
    normalized_x=True,
    categorical_y=True,
    is_torch=True,
)

loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=10, average="micro"),
        metric_wrapper(Precision, task="multiclass", num_classes=10, average="micro"),
    ],
)

device_list = [alice, bob]  # device list
server = charlie  # server
# aggregator = PlainAggregator(alice)
aggregator = SecurePruneAggregator(server, [alice, bob])

start_time = time.time()
# # training result with original model and strategy without pruning
# fl_model = FLModel(
#     server=server,
#     device_list=device_list,
#     model=model_def,
#     aggregator=aggregator, # secure aggregator
#     strategy='fed_avg_w',  # fl strategy
#     backend="torch",
# )
# history = fl_model.fit(
#     train_data,
#     train_label,
#     validation_data=(test_data, test_label),
#     epochs=10,
#     batch_size=32,
#     aggregate_freq=1,
# )
# plt.plot(history.global_history['multiclassaccuracy'])
# plt.plot(history.global_history['val_multiclassaccuracy'])
# plt.title('FLModel accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'], loc='upper left')
# plt.savefig('FL_Model_accuracy_convnet_20_32_mnist.jpg')
# plt.show()
#
# # training result with pruning model parameters
fl_model_prune = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy="fed_avg_w_prune",
    backend="torch",
    prune_end_rate=0.1,
    prune_percent=5,  # fix prune speed
)
history_prune = fl_model_prune.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=10,
    batch_size=32,
    aggregate_freq=1,
)
print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

# Draw accuracy values for training & validation when prune
plt.plot(history_prune.global_history["multiclassaccuracy"])
plt.plot(history_prune.global_history["val_multiclassaccuracy"])
plt.title("FLModel_prune accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Valid"], loc="upper left")
plt.savefig("./dgl/FL_prune_0.1_5_Model_accuracy_convnet_10_32_mnist.jpg")
plt.show()


# dgl attack verification
# tt = transforms.ToPILImage()
# img_index = 56  # image_index
# gt_data = tp(dst[img_index][0])  # true data
# gt_data = gt_data.view(1, *gt_data.size())
# gt_label = torch.Tensor([dst[img_index][1]]).long()  # true label
# gt_label = gt_label.view(
#     1,
# )
# gt_onehot_label = label_to_onehot(gt_label)  # one-hot label
# plt.imshow(tt(gt_data[0].cpu()))

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

gt_data = torch.load("./dgl/sf_output/x.pt")
gt_label = torch.load("./dgl/sf_output/y.pt")
tp = transforms.Compose([transforms.ToPILImage()])
plt.imshow(tp(gt_data[0].cpu()),cmap='gray')
plt.savefig("./dgl/true_to_valiate.jpg", dpi=300)
net = LeNet()  # original model
torch.manual_seed(1234)
net.apply(weights_init)  # init param
criterion = nn.CrossEntropyLoss()

tt = transforms.ToPILImage()
pred = net(gt_data)
y = criterion(pred, gt_label)
dy_dx = torch.autograd.grad(y, net.parameters())
# original dgl without pruning
original_dy_dx = list((_.detach().clone() for _ in dy_dx))  # true gradient

# generate dummy data and label
num_classes = 10
dummy_data = torch.randn(gt_data.size()).requires_grad_(True)
dummy_label = torch.randn((gt_data.shape[0], num_classes)).requires_grad_(True)
plt.imshow(tt(dummy_data[0].cpu()))
plt.savefig("./dgl/dummy.jpg", dpi=300)

# define dgl optimizer
LR = 1.0
ITERATION = 300
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

is_prune = False
if is_prune:
    original_dy_dx =[]
    for i in range(4):
        tens = torch.from_numpy(np.load("./dgl/sf_output/gradients" + str(i) + ".npy"))
        original_dy_dx.append(tens)

# dgl attack without pruning
history = []
history_iters = []
grad_difference = []
data_difference = []
current_loss = 0
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for iters in range(ITERATION):
    def closure():  # optimize dummy data, calculate dummy grad, grad diff
        optimizer.zero_grad()
        dummy_pred = net(dummy_data)
        dummy_loss = -torch.mean( # diff between pred dummy and dummy label
            torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(dummy_pred, -1)), dim=-1))
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # diff between true gradient and dummy gradient
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()  # backward gradient diff
        return grad_diff

    optimizer.step(closure)  # optimize closure (gradient diff)
    current_loss = closure().item()
    grad_difference.append(current_loss)
    data_difference.append(torch.mean((dummy_data - gt_data) ** 2).item())
    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
    print(current_time, iters, 'grad diff = %.8f, data diff = %.8f' % (current_loss, data_difference[-1]))
    ax1.plot(grad_difference)
    fig1.savefig("dgl/current_loss_" + str(LR) + "_" + "LBFGS" + ".jpg")
    ax2.plot(data_difference)
    fig2.savefig("dgl/data_difference_" + str(LR) + "_" + "LBFGS" + ".jpg")

    if iters % 10 == 0:  #
        history.append([tp(dummy_data[0].cpu())])
        history_iters.append(iters)

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 10, 1)
        plt.imshow(tp(gt_data[0].cpu()), cmap='gray')
        for i in range(min(len(history), 29)):
            plt.subplot(3, 10, i + 2)
            # plt.imshow(history[i][imidx])
            plt.imshow(history[i][0], cmap='gray')
            plt.title('iter=%d' % (history_iters[i]))
            plt.axis('off')
        if is_prune:
            plt.savefig("dgl/contrast_after_prune.jpg", dpi=300)
        else:
            plt.savefig("dgl/contrast_before_prune.jpg", dpi=300)
        plt.close()

