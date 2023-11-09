# 目的

想要实现一个自动化的流水线测试框架，能够提供简单的性能和时间报告。同时需要满足 components 之间的串联测试需求：

- 需要自动化的进行 {网络} x {数据集} x {组件} x {参数} x {协议} 组合的性能测试，并输出耗时时间/内存峰值。
- 能够进行pipeline串联测试，作为components自测，这里可能是 {组件} x {参数} 这个子集就能满足。

# 环境要求

在物理机/虚拟机的docker执行测试，并使用tc进行网络模拟。所以ssh账户需要有docker创建/销毁等权限，内核需要支持tc。

如果是跨多台物理机/虚拟机，则机器之间需要万兆互联。

需要互联网连接，使用yum。

# 整体设计

## 调度

使用ssh远程执行cmd进行节点控制，主要分3层：

- TestController代表中心端，下发命令给各个node(party)：初始环境/tc更改网络/执行comp等，遍历一遍所有的测试case，并统计测试结果。
- NodeController代表一个party执行结点，目前只支持运行在docker内，并通过docker的cgroup收集内存占用信息，远程调用节点调用comp.entry运行comp，收集comp运行结果等。每个party都拉起自己的local ray集群，party之间按照rayfed模式执行。

## 测试case

- TestVersion指定要测试哪些版本，可以是已经发布的公开版本，也可以是开发打包出来的whl
- TestNode指定测试的party要跑在哪里，docker的限制 / root目录在哪里（目前只支持local fs）/ ssh访问方式等
- NetCase 网络设定case，可配置带宽/延迟限制
- ClusterCase 测试协议/集群配置，是aby3 还是 N方的semi2k，这里配置
- DataCase 测试数据集，要在哪些数据(DAGInput)上跑这些测试，这些数据会最为pipeline中定义的DAG的根，这些数据集要放在TestNode的local fs目录中
- TestComp 定义单个comp，名称/参数
- PipelineCase 定义一个DAG，通过指定每个comp的上游，串联起来一个DAG，PSI等入口算子，指定TestData作为输入

# 示例

pipeline 串联测试见 tests/component/aci_pipeline.py 文件。

下面以开发中进行benchmark测试为例，首先：

```bash
# 创建sf dev whl包
python setup.py bdist_wheel
# 创建test_framework包
python secretflow/component/test_framework/setup.py bdist_wheel

# 推荐用conda创建一个独立的python环境继安装上述两个whl
```

然后编写测试脚本，并执行。benchmark脚本样例如下：

```python
from secretflow.component.test_framework.test_controller import TestController
from secretflow.component.data_utils import DistDataType
from secretflow.component.test_framework.test_case import (
    TestVersion,
    TestNode,
    TestComp,
    NetCase,
    ClusterCase,
    DAGInput,
    PipelineCase,
    DataCase,
)

if __name__ == '__main__':
    # 设定执行环境，这里设定 alice / bob 分别在36/35的docker中执行
    alice = TestNode(
        party="alice",
        local_fs_path="/root/tmp/alice",
        rayfed_port=62110,
        spu_port=62111,
        hostname="172.19.10.36",
        docker_cpu_limit=4,
        docker_mem_limit=0,
    )
    bob = TestNode(
        party="bob",
        local_fs_path="/root/tmp/bob",
        rayfed_port=62120,
        spu_port=62121,
        hostname="172.19.10.35",
        docker_cpu_limit=4,
        docker_mem_limit=0,
    )

    # 定义一个开发中版本，在whl上进行测试，whl_path需要能被本脚本访问到
    dev_version = TestVersion(
        "whl", whl_paths=["secretflow-0.8.2b3-cp38-cp38-manylinux2014_x86_64.whl"]
    )

    # 1000mb/1ms的网络设定
    net1 = NetCase("1000mb", 1000, 1)

    # 测试2方semi2k
    cluster1 = ClusterCase(
        "semi2k", ["alice", "bob"], "SEMI2K", "FM64", 18, ["alice", "bob"]
    )

    # 准备一组喂给PSI的单方数据集
    data_alice = DAGInput(
        DistDataType.INDIVIDUAL_TABLE,
        {"alice": "in.csv"},
        feature_columns={"alice": ['x1', 'x2', 'x3', 'x4', 'x5']},
        feature_types={"alice": ["float"] * 5},
        id_columns={"alice": ["id1"]},
        id_types = {"alice": ["str"]},
        label_columns={"alice": ["y"]},
        label_types = {"alice": ["float"]},
    )
    data_bob = DAGInput(
        DistDataType.INDIVIDUAL_TABLE,
        {"bob": "in.csv"},
        feature_columns={"bob": ['x6', 'x7', 'x8', 'x9', 'x10']},
        feature_types={"bob": ["float"] * 5},
        id_columns={"bob": ["id2"]},
        id_types = {"bob": ["str"]},
    )
    # 将这一组数据作为一个 data case
    data_case1 = DataCase("demo_case", {"alice": data_alice, "bob": data_bob})

    # 新增一个demo pipe
    demo_pipe = PipelineCase("demo_pipe")

    # 定义 psi 组件
    attrs = {
        "protocol": "ECDH_PSI_2PC",
        "receiver": "alice",
        "precheck_input": True,
        "sort": True,
        "broadcast_result": True,
        "bucket_size": 1048576,
        "ecdh_curve_type": "CURVE_FOURQ",
        "input/receiver_input/key": ["id1"],
        "input/sender_input/key": ["id2"],
    }
    psi = TestComp("psi_test", "preprocessing", "psi", "0.0.1", attrs)
    # 将psi增加到pipe中，并使用DataCase中准备的DAGInput.xx单方数据集作为输入，这里是整个DAG的起点
    demo_pipe.add_comp(psi, ["DAGInput.alice", "DAGInput.bob"])

    # 定义一个特征过滤，去掉x1/x9列
    attrs = {
        "input/in_ds/drop_features": ["x1", "x9"],
    }
    feature_filter = TestComp("ff", "preprocessing", "feature_filter", "0.0.1", attrs)
    # 添加进pipe， 使用 psi的第0个输出作为输入
    demo_pipe.add_comp(feature_filter, ["psi_test.0"])

    # 定义数据集分割
    attrs = {
        "train_size": 0.7,
        "test_size": 0.3,
        "random_state": 42,
        "shuffle": False,
    }
    ds_split = TestComp("ds_split", "preprocessing", "train_test_split", "0.0.1", attrs)
    # 添加进pipe，使用特征过滤的第0个输出作为输入
    demo_pipe.add_comp(ds_split, ["ff.0"])

    # 定义sslr 训练
    attrs = {
        "epochs": 2,
        "learning_rate": 0.1,
        "batch_size": 512,
        "sig_type": "t1",
        "reg_type": "logistic",
    }
    sslr = TestComp("sslr_train", "ml.train", "ss_sgd_train", "0.0.1", attrs)
    # 添加进pipe，使用数据集分割的第0个输出作为输入
    demo_pipe.add_comp(sslr, ["ds_split.0"])

    # 定义sslr预测
    attrs = {
        "batch_size": 2048,
        "receiver": "alice",
        "save_ids": True,
        "save_label": True,
    }
    sslr = TestComp("sslr_pred", "ml.predict", "ss_sgd_predict", "0.0.1", attrs)
    # 添加进pipe，使用sslr训练的第0个输出 和 数据集分割的第1个输出作为输入
    demo_pipe.add_comp(sslr, ["sslr_train.0", "ds_split.1"])

    # 开始测试
    test = TestController()
    # 在开发版本上进行测试
    test.add_test_version(dev_version)
    # 在 alice / bob上测试
    test.add_node(alice)
    test.add_node(bob)
    # 测试 1000mb网络 case
    test.add_net_case(net1)
    # 测试semi2k
    test.add_cluster_case(cluster1)
    # 测试数据集
    test.add_data_case(data_case1)
    # 测试pipe
    test.add_pipeline_case(demo_pipe)

    report = test.run(True)

    print(f"report:\n.....\n{report}\n......")


```

输出结果

```json
{'secretflow-0.8.2b3-cp38-cp38-manylinux2014_x86_64.whl': {'1000mb': {'semi2k': {'demo_case': {'demo_pipe': {'psi_test': BenchmarkRecord(test_name='psi_test', mem_peak=2.5462799072265625, run_time=2.102956533432007, status='finished'), 'ff': BenchmarkRecord(test_name='ff', mem_peak=2.5372543334960938, run_time=1.6035594940185547, status='finished'), 'ds_split': BenchmarkRecord(test_name='ds_split', mem_peak=2.5273590087890625, run_time=0.05517768859863281, status='finished'), 'sslr_train': BenchmarkRecord(test_name='sslr_train', mem_peak=3.2712364196777344, run_time=2.2352802753448486, status='finished'), 'sslr_pred': BenchmarkRecord(test_name='sslr_pred', mem_peak=2.873065948486328, run_time=1.009974479675293, status='finished')}}}}}}
```
