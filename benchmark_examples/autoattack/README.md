# 联邦攻防Benchmark

联邦攻防框架提供了自动调优工具（Secretflow Tuner），除了可以完成传统的automl能力，
还配合联邦学习的callback机制，实现攻防的自动调优。

用户实现攻击算法后，可以便捷地通过攻防框架，调整找到最合适的攻击参数、模型/数据拆分方法等，
可以借此来判断联邦算法的安全性。

在联邦算法的基础上，我们在几个经典的数据集+模型上，分别实现了几个攻击算法，并实现了benchmark，获得调优的结果。
目前支持的benchmark包括：

|    | datasets   | models   | defenses          | exploit   | fia       | fsha      | grad_lia  | lia       | norm      | replace   | replay    | batch_lia |
|---:|:-----------|:---------|:------------------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
|  0 | drive      | dnn      | no_defense        | -         | supported | -         | -         | -         | -         | -         | -         | supported |
|  1 | drive      | dnn      | de_identification | -         | supported | -         | -         | -         | -         | -         | -         | -         |
|  2 | drive      | dnn      | fed_pass          | -         | supported | -         | -         | -         | -         | -         | -         | supported |
|  3 | drive      | dnn      | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
|  4 | drive      | dnn      | mid               | -         | supported | -         | -         | -         | -         | -         | -         | -         |
|  5 | drive      | dnn      | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | supported |
|  6 | creditcard | dnn      | no_defense        | supported | -         | -         | supported | -         | supported | -         | -         | -         |
|  7 | creditcard | dnn      | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
|  8 | creditcard | dnn      | fed_pass          | supported | -         | -         | supported | -         | supported | -         | -         | -         |
|  9 | creditcard | dnn      | grad_avg          | supported | -         | -         | supported | -         | supported | -         | -         | -         |
| 10 | creditcard | dnn      | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 11 | creditcard | dnn      | mixup             | supported | -         | -         | supported | -         | supported | -         | -         | -         |
| 12 | bank       | dnn      | no_defense        | supported | supported | supported | supported | supported | supported | supported | supported | -         |
| 13 | bank       | dnn      | de_identification | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 14 | bank       | dnn      | fed_pass          | supported | supported | supported | supported | supported | supported | -         | -         | -         |
| 15 | bank       | dnn      | grad_avg          | supported | -         | supported | supported | supported | supported | -         | -         | -         |
| 16 | bank       | dnn      | mid               | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 17 | bank       | dnn      | mixup             | supported | -         | supported | supported | supported | supported | -         | -         | -         |
| 18 | bank       | deepfm   | no_defense        | -         | -         | -         | -         | -         | supported | supported | supported | -         |
| 19 | bank       | deepfm   | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 20 | bank       | deepfm   | fed_pass          | -         | -         | -         | -         | -         | supported | -         | -         | -         |
| 21 | bank       | deepfm   | grad_avg          | -         | -         | -         | -         | -         | supported | -         | -         | -         |
| 22 | bank       | deepfm   | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 23 | bank       | deepfm   | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 24 | movielens  | dnn      | no_defense        | -         | -         | -         | -         | -         | -         | supported | supported | -         |
| 25 | movielens  | dnn      | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 26 | movielens  | dnn      | fed_pass          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 27 | movielens  | dnn      | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 28 | movielens  | dnn      | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 29 | movielens  | dnn      | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 30 | movielens  | deepfm   | no_defense        | -         | -         | -         | -         | -         | -         | supported | supported | -         |
| 31 | movielens  | deepfm   | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 32 | movielens  | deepfm   | fed_pass          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 33 | movielens  | deepfm   | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 34 | movielens  | deepfm   | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 35 | movielens  | deepfm   | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 36 | criteo     | dnn      | no_defense        | supported | -         | supported | supported | -         | supported | supported | supported | supported |
| 37 | criteo     | dnn      | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 38 | criteo     | dnn      | fed_pass          | supported | -         | supported | supported | -         | supported | -         | -         | -         |
| 39 | criteo     | dnn      | grad_avg          | supported | -         | supported | supported | -         | supported | -         | -         | -         |
| 40 | criteo     | dnn      | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 41 | criteo     | dnn      | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 42 | criteo     | deepfm   | no_defense        | -         | -         | -         | -         | -         | supported | supported | supported | supported |
| 43 | criteo     | deepfm   | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 44 | criteo     | deepfm   | fed_pass          | -         | -         | -         | -         | -         | supported | -         | -         | -         |
| 45 | criteo     | deepfm   | grad_avg          | -         | -         | -         | -         | -         | supported | -         | -         | -         |
| 46 | criteo     | deepfm   | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 47 | criteo     | deepfm   | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 48 | mnist      | vgg16    | no_defense        | -         | supported | -         | supported | supported | -         | supported | supported | supported |
| 49 | mnist      | vgg16    | de_identification | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 50 | mnist      | vgg16    | fed_pass          | -         | supported | -         | supported | supported | -         | -         | -         | supported |
| 51 | mnist      | vgg16    | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 52 | mnist      | vgg16    | mid               | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 53 | mnist      | vgg16    | mixup             | -         | -         | -         | supported | supported | -         | -         | -         | supported |
| 54 | mnist      | resnet18 | no_defense        | -         | supported | -         | supported | supported | -         | supported | supported | supported |
| 55 | mnist      | resnet18 | de_identification | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 56 | mnist      | resnet18 | fed_pass          | -         | supported | -         | supported | supported | -         | -         | -         | supported |
| 57 | mnist      | resnet18 | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 58 | mnist      | resnet18 | mid               | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 59 | mnist      | resnet18 | mixup             | -         | -         | -         | supported | supported | -         | -         | -         | supported |
| 60 | cifar10    | vgg16    | no_defense        | -         | supported | -         | supported | supported | -         | supported | supported | supported |
| 61 | cifar10    | vgg16    | de_identification | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 62 | cifar10    | vgg16    | fed_pass          | -         | supported | -         | supported | supported | -         | -         | -         | supported |
| 63 | cifar10    | vgg16    | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 64 | cifar10    | vgg16    | mid               | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 65 | cifar10    | vgg16    | mixup             | -         | -         | -         | supported | supported | -         | -         | -         | supported |
| 66 | cifar10    | resnet20 | no_defense        | -         | -         | -         | -         | supported | -         | supported | supported | supported |
| 67 | cifar10    | resnet20 | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 68 | cifar10    | resnet20 | fed_pass          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 69 | cifar10    | resnet20 | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 70 | cifar10    | resnet20 | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 71 | cifar10    | resnet20 | mixup             | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 72 | cifar10    | resnet18 | no_defense        | -         | supported | -         | supported | supported | -         | supported | supported | supported |
| 73 | cifar10    | resnet18 | de_identification | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 74 | cifar10    | resnet18 | fed_pass          | -         | supported | -         | supported | supported | -         | -         | -         | supported |
| 75 | cifar10    | resnet18 | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 76 | cifar10    | resnet18 | mid               | -         | supported | -         | -         | -         | -         | -         | -         | -         |
| 77 | cifar10    | resnet18 | mixup             | -         | -         | -         | supported | supported | -         | -         | -         | supported |
| 78 | cifar10    | cnn      | no_defense        | -         | -         | -         | supported | supported | -         | -         | -         | supported |
| 79 | cifar10    | cnn      | de_identification | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 80 | cifar10    | cnn      | fed_pass          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 81 | cifar10    | cnn      | grad_avg          | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 82 | cifar10    | cnn      | mid               | -         | -         | -         | -         | -         | -         | -         | -         | -         |
| 83 | cifar10    | cnn      | mixup             | -         | -         | -         | supported | supported | -         | -         | -         | supported |
## 如何添加新的实现
代码在`benchmark_example/autoattack`目录下。

`applications`目录下为具体的数据集+模型实现。
其中目录结构为`数据集分类/数据集名称/模型名称/具体实现`，例如`image/cifar10/vgg16`。
请继承自`ApplicationBase`并实现其所需的接口，并在最下层的`__init__.py`中，将主类重命名为`App`（参考已有实现）。

`attacks`目录下为具体的攻击实现。
请继承自`AttackBase`，并实现其中的抽象方法。
Attack中编写的是通用的攻击代码，如果攻击依赖于具体的数据集，例如需要辅助数据集和辅助模型，
则在application下的数据+模型代码中，提供这些代码。

`defenses`目录下位具体的防御实现。
请继承自`DefenseBase`，并实现其中的抽象方法。

`config.yaml`文件是对实验的配置，以及超参的配置，如果你的算法需要调优超参，记得在这里添加超参搜索空间。

## 运行单条测试

`benchmark_example/autoattack/main.py`提供了单个应用的测试入口，其必须参数为数据集、模型、攻击。
并可通过可选参数，控制是否使用GPU、是否简化测试等选项。
通过如下命令查看选项帮助：

```shell
cd secretflow
python benchmark_example/autoattack/main.py --help
```

以bank数据集，dnn数据集为例，可以通过`main.py`完成训练、攻击或自动调优攻击。
支持的数据集和攻击如上述的表格所示。

```shell
cd secretflow
# 训练
python benchmark_example/autoattack/main.py bank dnn train
# 攻击
python benchmark_example/autoattack/main.py bank dnn lia
# 攻击 + 防御
python benchmark_example/autoattack/main.py bank dnn grad_avg
# 攻击 + 防御 + 自动调优
python benchmark_example/autoattack/main.py bank dnn lia grad_avg --auto --config="path/to/config"
```

## 运行benchmark

benchmark脚本提供了全部数据集下进行自动攻击调优的benchmark能力，并可以借助ray，实现gpu+分布式训练加速。

### 启动集群

benchmark脚本支持在单台机器上自动启动ray集群进行调优测试，
但通常其使用场景是在多台GPU机器下启动分布式ray集群，以加速实验。

首先需要参考Secretflow部署，在多台机器上完成conda环境安装，python环境安装，以及Secretflow的部署。
注意python版本必须一致。

我们首先要在机器上启动ray集群，并指定机器资源，其中包括：

- --gpu_nums选项：如果使用GPU，需要指定每台机器有几块GPU；
- 角色标签资源：如'alice'、'bob'，值和cpu数量相等即可；
- gpu内存资源：通过'gpu_mem'指定，单位为B，指定为该机器GPU总内存。


```shell
# 在首台机器上，启动ray头结点
ray start --head --port=6379 --resources='{"alice": 16, "bob":16, "gpu_mem": 85899345920}' --num-gpus=1 --disable-usage-stats --include-dashboard False
# 在其余机器上，启动ray并连接头结点
ray start --address="headip:6379" --resources='{"alice": 16, "bob":16, "gpu_mem": 85899345920}' --num-gpus=1 --disable-usage-stats
# 在头结点查看ray集群状态，看节点数量是否正确
ray status
```
### 启动benchmark

由于配置参数较多，通常可以通过指定配置文件(`config.yaml`)的方式来运行，配置文件格式和介绍如下：

```yaml
# application configurations.
applications:
  # enable auto tune or not.
  enable_tune: true
  # which dataset to run (all/bank/cifar10/[bank, drive, ...]/...)
  dataset: all
  # which model to run (all/dnn/deepfm/[dnn,vgg16,...]/...)
  model: all
  # which attack to run (all/lia/fia/[no_attack,lia,...]/...)
  attack: all
  # which defenses to run (all/grad_avg/[no_defense,mid,mixup,...]/...)
  defense: all
  # whether to run a simple test with small dataset and small epoch.
  simple: false
  # whether to use gpu to accelerate.
  use_gpu: false
  # whether using debug mode to run sf or not
  debug_mode: true
  # a random seed can be set to achieve reproducibility
  random_seed: ~
# path configurations.
paths:
  # the dataset store path, you can put the datasets here, or will auto download.
  datasets: ~
  # the autoattack result store path, default to '~/.secretflow/workspace'.
  autoattack_path: ~
# Resources configurations.
# Only needed when using sim mode and need to indicate the cpu/gpu nums manually.
resources:
  # how many CPUs do all your machines add up to.
  num_cpus: ~
  # how many CPUs do all your machines add up to (need applications.use_gpu = true).
  num_gpus: 2
# When there are multiple ray clusters in your machine, specify one to connect.
ray:
  # the existing ray cluster's address for connection (ip:port).
  address: ~
# tuner hyperparameters
tune:
  # application hyperparameters
  applications:
    # dataset
    creditcard:
      # model
      dnn:
        # all posible hyperparameters
        train_batch_size: [ 64, 128 ]
        hidden_size_range: [ 28, 64 ]
        alice_feature_nums_range: [ 25 ]
        dnn_base_units_size_range_alice: [ [ -0.5, -1 ],[ -1 ],[ -0.5, -1, -1 ], ]
        dnn_base_units_size_range_bob: [ [ 4 ] ]
        dnn_fuse_units_size_range: [ [ 1 ],[ -1, -1, 1 ], ]
        dnn_embedding_dim_range: ~
  # attack hyperparameters
  attacks:
    # attack name
    exploit:
      # all posible hyperparameters
      alpha_acc: [ 0.8,1 ] # 0 - 1
      alpha_grad: [ 0.01,0.1 ] # 0 -1  log
      alpha_kl: [ 0.01,0.1 ] # 0-1
    # when there is no hyperparameters, still add this attack.
    norm: ~
  # defense hyperparameters
  defenses:
    # defense name
    de_identification:
      # all posible hyperparameters
      subset_num: [ 3,5,7 ]
```

通过指定配置文件运行的方式如下，由于运行时间较长，建议使用nohup后台运行：

```shell
cd secretflow
# 如使用上述配置文件，会运行全部的case
nohup python benchmark_example/autoattack/benchmark.py --config="./benchmark_example/autoattack/config.yaml" > benchmark.log 2>&1 &
```

也可以通过命令行方式运行，具体的命令行请参考`--help`：

```shell
cd secretflow
# 帮助
python benchmark_example/autoattack/benchmark.py --help
# 指定dataset和gpu
python benchmark_example/autoattack/benchmark.py --dataset=all --use_gpu
```

如果命令行和配置文件同时指定，命令行添加的选项会覆盖配置文件的选项。

### 观察结果
Benchmark运行结束后，在`autoattack_path`指定的路径（默认为`~/.secretflow/workspace`）中，
可以找到benchmark启动时间的文件夹，其目录结构如下：

```python
<your autoattack path> (default to ~/.secretflow/workspace)
│
├── 2024-05-02-10-00-00/ # 启动时间命名的文件夹
│   ├── final_result_simple.md # 简化版的运行报告，可以快速看到最佳的运行结果和超参
└── ├── final_result.md # 详细的运行报告
    ├── bank_dnn_lia_mid/ # 以<dataset_model_attack_defense>命名的文件夹
    │   ├── best_result_acc_max.csv # 当前场景下，acc最大的结果
    │   ├── full_result.csv # 当前场景下的全部组合以及运行结果
    │   ├── xxxx/ # 其余文件可通过Tensorboard实现运行结果的可视化，参考https://docs.ray.io/en/latest/tune/tutorials/tune-output.html#how-to-log-your-tune-runs-to-tensorboard
    │   └── ...
    ├── .../ 其余场景的文件夹
```