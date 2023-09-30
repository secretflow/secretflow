# SplitGuard Detection
1. 原理
   
   Erdogan等人通过研究FSHA攻击，针对性提出了SplitGuard，一种拆分学习客户端可以检测其是否成为训练劫持攻击目标的方法。SplitGuard的出发点是如果客户的本地模型正在学习预期的分类任务，那么当任务被逆转时（即当原始任务的成功意味着新任务的失败时）它应该以截然不同的方式表现。

   [论文] https://arxiv.org/abs/2108.09052

2. 代码文件结构
    ``` python
    root:SplitGuard/

        slmodelsg.py #训练过程中SG score的计算实现
        mnist_sl_torch_splitguard.ipynb #测试样例

    ```
