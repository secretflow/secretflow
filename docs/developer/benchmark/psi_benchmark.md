# 隐语PSI Benchmark白皮书

> This tutorial is only available in Chinese.

## 导语
为了方便大家了解隐语的Benchmark，我们设计了10分钟上手手册，包含了亮点介绍、SecretFlow集群的易用搭建、Benchmark脚本、两方和三方的Benchmark，使相关业务方做调研时方便拿到可度量的性能数据和可复现的路径。

## 隐语PSI亮点
隐私集合求交（Private Set Intersection，简写为：PSI）是一类特定的安全多方计算（Multi-Party Computation, 即MPC）问题，其问题可以简单理解为：Alice 输入集合 X，Bob 输入集合 Y，双方执行 PSI 协议可以得到 Alice 和 Bob 两者的交集，同时不在交集范围内的部分是受保护的，即 Alice 和 Bob 无法学习出交集以外的任何信息。
隐私集合求交(PSI)协议有很多分类方法，按照底层依赖的密码技术分类主要包括：
- 基于公钥密码的PSI方案，包括：基于判定型密钥交换（DDH: Decisional Diffie-Hellman）的PSI方案和RSA盲签名的PSI方案；
- 基于不经意传输（OT: Oblivious Transfer）的PSI方案；
- 基于通用MPC的PSI方案，例如基于混淆电路（GC: Garbled Circuit）的PSI方案；
- 基于同态加密（Homomorphic Encryption）的PSI方案。
隐私集合求交(PSI)协议按照参与方的数量进行分类，可分为：
- 两方PSI：参与方为2个；
- 多方PSI：参与方>2个。
隐私集合求交(PSI)协议按照设定安全模型分类，可分为：
- 半诚实模型的PSI；
- 恶意模型的PSI。
SecretFlow SPU 实现了半诚实模型下的两方和三方PSI协议，密钥安全强度是128bit，统计安全参数是40bit。
- 两方PSI(Private Set Intersection)协议：
  - 基于DDH的PSI协议，
    - 基于DDH的PSI协议先对简单易于理解和实现，依赖的密码技术已被广泛论证，通信量低，但计算量较大。
    - 隐语实现了基于椭圆曲线(Elliptic Curve)群的DDH PSI协议，支持的椭圆曲线类型包括：Curve25519,SM2,Secp256k1。本次benchmark选用的曲线是Curve25519。
  - 基于OT扩展的KKRT16
    - KKRT16是第一个千万规模(224)求交时间在1分钟之内的PSI方案，通信量较大；
    - 隐语实现了KKRT16协议，并参考了进年来的性能优化和安全改进方案，例如：stash-less CuckooHash，[GKWW20]中 FixedKey AES作为 correlation-robust 哈希函数。
  - 基于PCG的BC22
    - BC22 PSI依赖的PCG(Pseudorandom Correlation Generator)方案是近年来mpc方向的研究热点，相比KKRT16计算量和通信两方面都有了很大改进，从成本(monetary cost)角度更能满足实际业务需求。PCG实现依赖LPN(Learning Parity with Noise)问题，由于是2022年最新的协议，协议的安全性还需要更多密码专家的分析和论文。
    - 隐语0.7中实现了BC22 PSI方案，其中的PCG/VOLE使用了emp-zk中的[WYKW21]实现，欢迎大家审查和进一步改进；
- 三方PSI(Private Set Intersection)协议：
  - 基于DDH的三方PSI协议
    - 隐语实现了自研的基于 ECDH 的三方 PSI 协议，注意我们实现的这个协议会泄漏两方交集大小，请自行判断是否满足使用场景的安全性，本次benchmark选用的曲线是Curve25519。
## 复现方式
### 一、测试机型环境
- Python：3.8
- pip: >= 19.3
- OS: CentOS 7
- CPU/Memory: 推荐最低配置是 8C16G
- 硬盘：500G
### 二、安装conda
使用conda管理python环境，如果机器没有conda需要先安装，步骤如下：
#sudo apt-get install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

#### 详细步骤
```
#sudo apt-get install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#安装
bash Miniconda3-latest-Linux-x86_64.sh
# 一直按回车然后输入yes
please answer 'yes' or 'no':
>>> yes
# 选择安装路径, 文件名前加点号表示隐藏文件
Miniconda3 will now be installed into this location:
>>> ~/.miniconda3
# 添加配置信息到 ~/.bashrc文件
Do you wish the installer to initialize Miniconda3 by running conda init? [yes|no]
[no] >>> yes
#运行配置信息文件
source ~/.bashrc
#测试是否安装成功
conda --version 
```
![](./resources/4bded9b0-d913-48b2-b7a9-c05e0d2c7c81.png)
         

### 三、安装secretflow
```
# 创建干净的python环境
conda create -n sf-benchmark python=3.8

# 进入benchmark 环境
conda activate sf-benchmark

# 安装secretflow
pip install -U secretflow

# 创建一个sf-benchmark目录
mkdir sf-benchmark
cd sf-benchmark
```

验证安装是否成功
root目录下输入python然后回车；
```
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], address='local')
>>> dev = sf.PYU('alice')
>>> import numpy as np
>>> data = dev(np.random.rand)(3, 4)
>>> sf.reveal(data)
```
如下图所示就代表环境搭建成功了
![](./resources/9bab546b-6578-4ff7-b8f7-9f26ab4df46a.png)

### 四、创建节点并启动集群
#### 创建ray header节点
创建ray header节点，选择一台机器为主机，在主机上执行如下命令，ip替换为主机的内网ip，命名为alice，端口选择一个空闲端口即可
注意：192.168.0.1 ip为mock的，请替换为实际的ip地址
```
RAY_DISABLE_REMOTE_CODE=true \
ray start --head --node-ip-address="192.168.0.1" --port="9394" --resources='{"alice": 8}' --include-dashboard=False
```
#### 创建从属节点
创建从属节点，在bob机器执行如下命令，ip依然填alice机器的内网ip，命名为bob，端口不变
```
RAY_DISABLE_REMOTE_CODE=true \
ray start --address="192.168.0.1:9394" --resources='{"bob": 8}'
```
创建从属节点，在carol机器执行如下命令，ip依然填alice机器的内网ip，命名为carol，端口不变
```
RAY_DISABLE_REMOTE_CODE=true \
ray start --address="192.168.0.1:9394" --resources='{"carol": 8}'
```
#### 验证节点是否启动
在python中测试节点是否启动成功，任意选一台机器输入python，执行下列代码，参数中address为头节点(alice)的地址，拿alice机器来验证，每输入一行下列代码回车一次：
```
>>> import secretflow as sf
>>> sf.init(['alice','bob'], address='192.168.0.1:9394')
>>> alice = sf.PYU('alice')
>>> bob = sf.PYU('bob')
>>> sf.reveal(alice(lambda x : x)(2))
>>> sf.reveal(bob(lambda x : x)(2))
```
如下图就代表节点创建成功了
![](./resources/3386cb76-53c1-4df6-ae5e-a26314609d5c.png)
同时我们也可以通过ray status去看节点的状态，前提是先进入sf环境（conda activate sf-benchmark）
![](./resources/e63ba232-025b-4b4f-9c57-d61d80cc8a1f.png)

#### 生成数据

把[generate_psi.py](https://github.com/secretflow/spu/blob/main/spu/psi/tools/generate_psi.py)脚本传到alice机器的root目录下
执行如下代码
```
# 生成三份一千万数据
python3 generate_psi.py 10000000

# 生成三份一亿数据
python3 generate_psi.py 100000000
```
把生成的psi_1.csv cp到benchmark目录下，再通过scp的命令把psi_2.csv/psi_3.csv分别移到bob的benchmark目录下跟carol的benchark目录下 

#### 限制宽带/延迟
```
#100Mbps 20ms
 tc qdisc add dev eth0 root handle 1: tbf rate 100mbit burst 256kb latency 800ms                                    
 tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 20msec limit 8000 

清除限制
tc qdisc del dev eth0 root
查看已有配置
tc qdisc show dev eth0
```
#### Benchmark脚本
支持的PSI协议列表：
- ECDH_PSI_2PC
- KKRT_PSI_2PC
- BC22_PSI_2PC
- ECDH_PSI_3PC
```
import sys
import time
import logging

from absl import app
import spu
import secretflow as sf

# init log
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# SPU settings
cluster_def = {
    'nodes': [
        # <<< !!! >>> replace <192.168.0.1:12945> to alice node's local ip & free port
        {'party': 'alice', 'address': '192.168.0.1:12945', 'listen_address': '0.0.0.0:12945'},
        # <<< !!! >>> replace <192.168.0.2:12946> to bob node's local ip & free port
        {'party': 'bob', 'address': '192.168.0.2:12946', 'listen_address': '0.0.0.0:12946'},
        # <<< !!! >>> if you need 3pc test, please add node here, for example, add carol as rank 2
        # {'party': 'carol', 'address': '127.0.0.1:12347'},
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
    },
}


def main(_):

    # sf init
    # <<< !!! >>> replace <192.168.0.1:9394> to your ray head
    sf.init(['alice','bob'], address='192.168.0.1:9394')
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')

    # <<< !!! >>> replace path to real parties local file path.
    input_path = {
        alice: '/data/psi_1.csv',
        bob: '/data/psi_2.csv',
        # if run with `ECDH_PSI_3PC`, add carol
        # carol: '/data/psi_3.csv',
    }
    output_path = {
        alice: '/data/psi_output.csv',
        bob: '/data/psi_output.csv',
        # if run with `ECDH_PSI_3PC`, add carol
        # carol: '/data/psi_output.csv',
    }
    select_keys = {
        alice: ['id'],
        bob: ['id'],
        # if run with `ECDH_PSI_3PC`, add carol
        # carol: ['id'],
    }
    spu = sf.SPU(cluster_def)

    # prepare data
    start = time.time()

    reports = spu.psi_csv(
        key=select_keys,
        input_path=input_path,
        output_path=output_path,
        receiver='alice',  # if `broadcast_result=False`, only receiver can get output file.
        protocol='KKRT_PSI_2PC',	# psi protocol
        precheck_input=False,  # will cost ext time if set True
        sort=False,  # will cost ext time if set True
        broadcast_result=False,  # will cost ext time if set True
    )
    print(f"psi reports: {reports}")
    logging.info(f"cost time: {time.time() - start}")

    sf.shutdown()


if __name__ == '__main__':
    app.run(main)
```


### 五、Benchmark报告
![](./resources/7629c228-bc51-4ef7-93e9-9f0c465d025d.png)
目前bechmark数据中，bc22 psi的性能还在进一步工程优化， 单测spu中bc22协议内核的性能对比可以参考
[pcg psi](https://mp.weixin.qq.com/s?__biz=MzA5NTQ0MTI4OA==&mid=2456927355&idx=1&sn=832269f138e35f031bc2bdcd63f05520&chksm=873a449cb04dcd8a4dacd4cec0ccc7c147219a76f36a6d694f26b7c2a27d03be8f968578fab4&scene=21#wechat_redirect)的介绍。

- ECDH：对网络配置不敏感，对计算资源敏感，通常用于多方数据量均衡时，带宽在100M及以下时计算速度比KKRT快，带宽小于等于100M时推荐；
- KKRT：网络设置为100Mbps时，带宽成为瓶颈。通常用于两方数据量均衡时，高带宽时计算速度快，带宽大于等于1000M时推荐；







        
