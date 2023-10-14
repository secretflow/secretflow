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
# dgl attack without pruning
# dst = datasets.CIFAR100("./data", download=True)
# tp = transforms.ToTensor()
tt = transforms.ToPILImage()
# img_index = 56  # image_index
# gt_data = tp(dst[img_index][0])  # true data
# gt_data = gt_data.view(1, *gt_data.size())
# gt_label = torch.Tensor([dst[img_index][1]]).long()  # true label
# gt_label = gt_label.view(
#     1,
# )
# gt_onehot_label = label_to_onehot(gt_label)  # one-hot label
# plt.imshow(tt(gt_data[0].cpu()))


net = LeNet()  # original model
torch.manual_seed(1234)

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

gt_data = torch.load("./dgl/sf_output/x.pt")
gt_label = torch.load("./dgl/sf_output/y.pt")
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
plt.savefig("./dgl/true_to_valiate.jpg", dpi=300)
net.apply(weights_init)  # init param
criterion = cross_entropy_for_onehot  # criterion
pred = net(gt_data)
gt_onehot_label = label_to_onehot(gt_label)  # one-hot label
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))  # true gradient
print(original_dy_dx)

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).requires_grad_(True)
plt.imshow(tt(dummy_data[0].cpu()))
plt.savefig("./dgl/dummy.jpg", dpi=300)

# define dgl optimizer
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

is_prune = True
if is_prune:
    for i in range(4):
        tens = torch.from_numpy(np.load("./dgl/sf_output/gradients" + str(i) + ".npy"))
        original_dy_dx.append(tens)

# dgl attack without pruning
history = []
for iters in range(300):

    def closure():  # optimize dummy data, calculate dummy grad, grad diff
        optimizer.zero_grad()
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(
            dummy_pred, dummy_onehot_label
        )  # diff between pred dummy and dummy label
        dummy_dy_dx = torch.autograd.grad(
            dummy_loss, net.parameters(), create_graph=True
        )  # dummy gradient
        grad_diff = 0
        for gx, gy in zip(
            dummy_dy_dx, original_dy_dx
        ):  # diff between true gradient and dummy gradient
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()  # backward gradient diff
        return grad_diff

    optimizer.step(closure)  # optimize closure (gradient diff)
    if iters % 10 == 0:  #
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

# plot the process of dummy data
plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis("off")
# plt.savefig("contrast_before_prune.jpg", dpi=300)
plt.savefig("./dgl/contrast_after_prune.jpg", dpi=300)
