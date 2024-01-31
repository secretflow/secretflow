# 联邦攻防Benchmark

联邦攻防框架提供了自动调优工具（secretflow tuner），除了可以完成传统的automl能力，
还配合联邦学习的callback机制，实现攻防的自动调优。

用户实现攻击算法后，可以便捷地通过攻防框架，调整找到最合适的攻击参数、模型/数据拆分方法等，
可以借此来判断联邦算法的安全性。

在联邦算法的基础上，我们在几个经典的数据集+模型上，分别实现了几个攻击算法，并实现了benchmark，获得调优的结果。
目前支持的benchmark包括：

|    | datasets  | models   | lia | fia | replay | replace | exploit | norm |
|---:|:----------|:---------|:----|:----|:-------|:--------|:--------|:-----|
|  0 | bank      | dnn      | ✅   |     | -      | -       | -       | ✅    |
|  1 | bank      | deepfm   |     |     | -      | -       | -       | ✅    |
|  2 | bank      | resnet18 | -   | -   | -      | -       | -       | -    |
|  3 | bank      | vgg16    | -   | -   | -      | -       | -       | -    |
|  4 | bank      | resnet20 | -   | -   | -      | -       | -       | -    |
|  5 | movielens | dnn      |     |     | -      | -       | -       | -    |
|  6 | movielens | deepfm   |     |     | -      | -       | -       | -    |
|  7 | movielens | resnet18 | -   | -   | -      | -       | -       | -    |
|  8 | movielens | vgg16    | -   | -   | -      | -       | -       | -    |
|  9 | movielens | resnet20 | -   | -   | -      | -       | -       | -    |
| 10 | drive     | dnn      | -   | ✅   | -      | -       | -       | -    |
| 11 | drive     | deepfm   | -   | -   | -      | -       | -       | -    |
| 12 | drive     | resnet18 | -   | -   | -      | -       | -       | -    |
| 13 | drive     | vgg16    | -   | -   | -      | -       | -       | -    |
| 14 | drive     | resnet20 | -   | -   | -      | -       | -       | -    |
| 15 | criteo    | dnn      | -   | -   | -      | -       | -       | ✅    |
| 16 | criteo    | deepfm   | -   | -   | -      | -       | -       | ✅    |
| 17 | criteo    | resnet18 | -   | -   | -      | -       | -       | -    |
| 18 | criteo    | vgg16    | -   | -   | -      | -       | -       | -    |
| 19 | criteo    | resnet20 | -   | -   | -      | -       | -       | -    |
| 20 | mnist     | dnn      | -   | -   | -      | -       | -       | -    |
| 21 | mnist     | deepfm   | -   | -   | -      | -       | -       | -    |
| 22 | mnist     | resnet18 | -   | -   | -      | -       | -       | -    |
| 23 | mnist     | vgg16    | -   | -   | -      | -       | -       | -    |
| 24 | mnist     | resnet20 | -   | -   | -      | -       | -       | -    |
| 25 | cifar10   | dnn      | -   | -   | -      | -       | -       | -    |
| 26 | cifar10   | deepfm   | -   | -   | -      | -       | -       | -    |
| 27 | cifar10   | resnet18 | ✅   | -   | -      | -       | -       | -    |
| 28 | cifar10   | vgg16    | -   | -   | -      | -       | -       | -    |
| 29 | cifar10   | resnet20 | ✅   | -   | -      | -       | -       | -    |

## 如何添加新的实现

代码在`benchmark_example/autoattack`目录下。

`applications`目录下为具体的数据集+模型实现。
其中目录结构为`数据集分类/数据集名称/模型名称/具体实现`，例如`image/cifar10/vgg16`。

`attacks`目录下为具体的攻击实现。
其中编写的是通用的攻击代码，如果攻击依赖于具体的数据集，例如需要辅助数据集和辅助模型，则在application下的数据+模型代码中，提供这些代码。

ps：需要在`autoattack/utils/distribution.py`中添加新的实现，以便能够检索到。


## 运行单条测试

```shell
cd secretflow
# 训练
python benchmark_example/autoattack/main.py bank dnn train
# 攻击
python benchmark_example/autoattack/main.py bank dnn lia
# auto攻击
python benchmark_example/autoattack/main.py bank dnn auto_lia
```

## 运行benchmark

```shell
cd secretflow
# 训练
python benchmark_example/autoattack/benchmark.py train
# auto
python benchmark_example/autoattack/benchmark.py auto
```