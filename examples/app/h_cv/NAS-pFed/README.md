# 介绍
本仓库是NAS-enabled Personalized Federated Learning for Heterogeneous Data算法的实现，通过结合神经架构搜索与知识蒸馏帮助每个客户端搜索更适合其数据分布的模型架构，然后通过本地训练获取性能更好的模型。

# 使用
1. 客户端和服务器定义：分别位于model文件夹下的Client.py和Server文件
2. 数据集分割：分为位于dataSplit文件夹下的cifar10、cifar100、mnist文件夹
3. 训练启动入口：根目录下的cifar10.py、cifar100.py、mnist.py文件，分别用于启动cifar10数据集、cifar100数据集、mnist数据集的训练
4. 结果：训练完毕后，准确率会保存在result目录中

# 参考仓库
* https://github.com/Astuary/Flow.git
* https://github.com/chaoyanghe/FedNAS.git