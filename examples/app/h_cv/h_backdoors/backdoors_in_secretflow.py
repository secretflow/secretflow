import secretflow as sf

print('The version of SecretFlow: {}'.format(sf.__version__))

sf.shutdown()


sf.init(['alice', 'bob', 'charlie'], address='local')
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

from secretflow.ml.nn.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision
from secretflow.security.aggregation import SecureAggregator
from secretflow.utils.simulation.datasets import load_mnist
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出
class CNNet(BaseModule):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNNet, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output

def poison_train_data_MNIST(train_data, train_label, poison_ratio:float, target_label:int):
    assert  poison_ratio < 0.2 ## 污染比例不可过大
    ##  train_data 样本  :[ ( img:np.array 1,28,28 ) ]

    ## 创建随机后门pattern
    pattern = np.zeros((1, 28, 28))
    for i in range(28):
        for j in range(28):
            pattern[0, i, j] = np.random.randint(0, 255)
            
    ## 将部分样本进行后门注入攻击。
    samples_num = len(train_data) ## label = 0 的样本的数量
    poison_num = int(samples_num * poison_ratio )  ## 污染数量
    ## 对数据进行 污染
    for i in range(poison_num):
        train_label[ int(i/poison_ratio) ] = target_label ## 篡改标签
        train_data[ int(i/poison_ratio) ] = pattern ## 篡改图像数据
    return train_data ,train_label



data = np.load("./mnist.npz")
train_data, train_label = data['x_train'], data['y_train']
test_data, test_label = data['x_test'], data['y_test']

train_data, train_label = poison_train_data_MNIST(train_data, train_label,0.1,8)

alice_train_data = train_data[:30000]
alice_test_data = test_data[:30000]
alice_train_label = train_label[:30000]
alice_test_label = test_label[:30000]

bob_train_data = train_data[30000:]
bob_test_data = test_data[30000:]
bob_train_label = train_label[30000:]
bob_test_label = test_label[30000:]

np.savez(
    "./alice_mnist.npz",
    train_x=alice_train_data,
    test_x=alice_test_data,
    train_y=alice_train_label,
    test_y=alice_test_label,
)
np.savez(
    "./bob_mnist.npz",
    train_x=bob_train_data,
    test_x=bob_test_data,
    train_y=bob_train_label,
    test_y=bob_test_label,
)

alice_path = "./alice_mnist.npz"
bob_path = "./bob_mnist.npz"

from secretflow.data.ndarray import load
from secretflow.data.split import train_test_split

fed_data = load({alice: alice_path, bob: bob_path}, allow_pickle=True)

(fed_data["train_x"], fed_data["train_y"]), (fed_data["test_x"], fed_data["test_y"]) = load_mnist(
    parts={alice: 0.5, bob: 0.5},
    normalized_x=True,
    categorical_y=True,
    is_torch=True,
)

loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
model_def = TorchModel(
    model_fn=CNNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=10, average='micro'),
        metric_wrapper(Precision, task="multiclass", num_classes=10, average='micro'),
    ],
)

device_list = [alice, bob]
server = charlie
aggregator = SecureAggregator(server, [alice, bob])

# 参数设定
fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy='fed_avg_w',  # fl strategy
    backend="torch",  # backend support ['tensorflow', 'torch']
)

history = fl_model.fit(
    fed_data["train_x"],
    fed_data["train_y"],
    validation_data=(fed_data["test_x"], fed_data["test_y"]),
    epochs=20,
    batch_size=32,
    aggregate_freq=1,
)

from matplotlib import pyplot as plt

# Draw accuracy values for training & validation
plt.plot(history.global_history['multiclassaccuracy'])
plt.plot(history.global_history['val_multiclassaccuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('FLmodel_accuracy.jpg')