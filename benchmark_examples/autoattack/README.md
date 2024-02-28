# 联邦攻防Benchmark

联邦攻防框架提供了自动调优工具（secretflow tuner），除了可以完成传统的automl能力，
还配合联邦学习的callback机制，实现攻防的自动调优。

用户实现攻击算法后，可以便捷地通过攻防框架，调整找到最合适的攻击参数、模型/数据拆分方法等，
可以借此来判断联邦算法的安全性。

在联邦算法的基础上，我们在几个经典的数据集+模型上，分别实现了几个攻击算法，并实现了benchmark，获得调优的结果。
目前支持的benchmark包括：

|    | datasets   | models   | auto_exploit | auto_fia | auto_lia | auto_norm | auto_replace | auto_replay |
|---:|:-----------|:---------|:-------------|:---------|:---------|:----------|:-------------|:------------|
|  0 | creditcard | dnn      | Support      | -        | -        | Support   | -            | -           |
|  1 | bank       | dnn      | Support      | Support  | Support  | Support   | Support      | Support     |
|  2 | bank       | deepfm   | -            | -        | -        | Support   | Support      | Support     |
|  3 | movielens  | dnn      | -            | -        | -        | -         | Support      | Support     |
|  4 | movielens  | deepfm   | -            | -        | -        | -         | Support      | Support     |
|  5 | criteo     | dnn      | Support      | -        | -        | Support   | Support      | Support     |
|  6 | criteo     | deepfm   | -            | -        | -        | Support   | Support      | Support     |
|  7 | mnist      | vgg16    | -            | Support  | Support  | -         | Support      | Support     |
|  8 | mnist      | resnet18 | -            | Support  | Support  | -         | Support      | Support     |
|  9 | drive      | dnn      | -            | Support  | -        | -         | -            | -           |
| 10 | cifar10    | vgg16    | -            | Support  | Support  | -         | Support      | Support     |
| 11 | cifar10    | resnet20 | -            | -        | Support  | -         | Support      | Support     |
| 12 | cifar10    | resnet18 | -            | Support  | Support  | -         | Support      | 3Support    |

## 如何添加新的实现

代码在`benchmark_example/autoattack`目录下。

`applications`目录下为具体的数据集+模型实现。
其中目录结构为`数据集分类/数据集名称/模型名称/具体实现`，例如`image/cifar10/vgg16`。
请继承自ApplicationBase并实现其所需的接口。

`attacks`目录下为具体的攻击实现。
其中编写的是通用的攻击代码，如果攻击依赖于具体的数据集，例如需要辅助数据集和辅助模型，则在application下的数据+模型代码中，提供这些代码。

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

benchmark脚本提供了运行全部数据集的能力，并可以借助ray，实现gpu+分布式训练加速。
通常情况可以通过指定配置文件的方式，来进行配置和运行，配置文件如下：

```yaml
# application configurations.
applications:
  # which target to run (train/attack/auto).
  mode: train
  # which dataset to run (all/bank/...)
  dataset: all
  # which model to run (all/bank/...)
  model: all
  # which attack to run (all/bank/...)
  attack: all
  # whether to run a simple test with small dataset and small epoch.
  simple: false
  # whether to use gpu to accelerate.
  use_gpu: false
  # whether using debug mode to run sf or not
  debug_mode: true
paths:
  # the dataset store path, you can put the datasets here, or will auto download.
  datasets: ~
  # the autoattack result store path.
  autoattack_path: ~
resources:
  # how many CPUs do all your machines add up to.
  num_cpus: ~
  # how many CPUs do all your machines add up to (need applications.use_gpu = true).
  num_gpus: ~

```

通过指定配置文件运行的方式如下：

```shell
cd secretflow
python benchmark_example/autoattack/benchmark.py --config="./benchmark_example/autoattack/config.yaml"
```

也可以通过命令行方式运行，具体的命令行请参考`-h`：

```shell
cd secretflow
python benchmark_example/autoattack/benchmark.py --dataset=all --use_gpu
```


如果命令行和配置文件同时指定，命令行添加的选项会覆盖配置文件的选项。