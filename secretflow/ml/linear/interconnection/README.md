# 隐私路由的开发指南

隐私路由旨在实现异构隐私计算平台间的数据互联互通。

目前支持同态加密（HE）和秘密共享（SS）数据之间的互相转化和流通。

隐私路由包括隐私路由客户端 Router Client（RC）和隐私路由服务端 Router Server（RS）两种组件。

RC负责收发隐私路由数据包，RS负责对隐私路由数据包的处理以及转发。

## 文件结构说明

- demo：隐私路由demo文件夹
  - he2ss：通过隐私路由，将同态加密数据转换成秘密共享数据分片的示例代码
  - ss2he：通过隐私路由，将秘密共享数据分片转换成同态加密数据的示例代码
  - horizontal_lr：通过隐私路由，实现隐语中横向LR（`tests/ml/linear/test_fl_lr_mix.py`）和其他隐私计算平台的横向lr的权重数据的互通
- deps：隐私路由依赖的python安装包
  - anyconn_core-0.0.5-py3-none-any.whl
  - router-0.1.0-py3-none-any.whl
- router_hooks.py：隐语`class FlLogisticRegressionMix`使用隐私路由进行数据交互的hook，其中的`WeightArbiter`使用RC进行权重的交互传输。

## 隐私路由demo

隐私路由的demo中包括了HE和SS数据互联互通的3种情况：

1. SS->HE：秘密共享数据分片转化为同态加密数据。
2. HE->SS：同态加密数据转化为秘密共享数据分片，其中同态加密的发送数据方有公钥和私钥，该过程可以认为是明文数据到秘密共享分片的转化。
3. HE->SS：同态加密数据转化为秘密共享数据分片，其中同态加密的发送数据方只有公钥。

**如果需要开发和隐语互联互通的隐私路由，可以参考如下demo**

### 0. 运行demo前的预安装

```python
pip install anyconn_core-0.0.5-py3-none-any.whl
pip install router-0.1.0-py3-none-any.whl
```

### Demo 1. SS->HE

SS->HE：秘密共享数据分片转化为同态加密数据。

![互联互通-SS2HE](/Users/chenlu/workspace/secretflow_chenlu/secretflow/ml/linear/interconnection/img/互联互通-SS2HE.png)

`demo/ss2he`目录下的5个python文件同时运行，可模拟上图SS2HE过程。

### Demo 2. HE->SS（HE发送方有公钥和私钥）

同态加密数据转化为秘密共享数据分片，其中同态加密的发送数据方有公钥和私钥，该过程可以认为是明文数据到秘密共享分片的转化。

![互联互通-HE2SS-plain](/Users/chenlu/workspace/secretflow_chenlu/secretflow/ml/linear/interconnection/img/互联互通-HE2SS-plain.png)

`demo/horizontal_lr`目录下的4个python文件同时运行，可模拟平台A和平台B横向lr过程中权重数据的交互，包括HE2SS和SS2HE，迭代交替运行。

### Demo 3. HE->SS（HE发送方只有公钥）

同态加密数据转化为秘密共享数据分片，其中同态加密的发送数据方只有公钥。

![互联互通-HE2SS](/Users/chenlu/workspace/secretflow_chenlu/secretflow/ml/linear/interconnection/img/互联互通-HE2SS.png)

`demo/he2ss`目录下的5个python文件同时运行，可模拟上图HE2SS过程。

注：在运行一次后，需要清理一下当前目录下生成的public_key文件。

## 与隐语横向LR互联互通的样例

已实现了在隐语`class FlLogisticRegressionMix`使用隐私路由进行数据交互的hook，即router_hooks.py。其中的`WeightArbiter`使用RC进行权重的交互传输。

样例为在基于ss的隐私计算平台（平台A）进行两边lr权重的聚合工作，聚合成功后，平台A使用聚合后参数进行下一轮迭代，secretflow收到平台A传回来的聚合后参数进行下一轮迭代。

### 运行SS侧代码

可以使用`horizontal_lr`下的`ss_rc_01.py`,`ss_rc_02.py`,`rs.py`三个文件模拟基于ss的隐私计算平台侧行为，和secretflow进行联调。

### 运行secretflow

取消`tests/ml/linear/test_fl_lr_mix.py`L127的注释：

```
aggr_hooks=[RouterLrAggrHook(devices.alice)],
```

在项目根目录下运行：

```
pytest tests/ml/linear/test_fl_lr_mix.py  --env=prod -v --capture=no
```
