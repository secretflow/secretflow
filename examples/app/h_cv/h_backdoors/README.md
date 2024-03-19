# 可行执行环境下FL数据后门攻击

## 项目介绍

本项目基于蚂蚁集团的secretflow隐语平台，利用隐语平台的联邦学习框架，对多种多样的后门攻击方法进行可信执行环境下的注入，推动后门攻击预防、检测、擦除等研究的进一步开展与深化。

## 项目运行
``` conda activate secretflow ```
``` python3 backdoors_in_secretflow.py ```

## 项目展望

- 学生针对cifar10，imagenet，pipa等数据集，应用poison_train_data函数，实现不同训练集上的投毒与检测。
- 学生通过对backdoor101中不同后门攻击算法的学习，开发后门攻击注入函数实现后门攻击。
- 学生设计resnet，lenet，googlenet等神经网络，在隐语平台上完成实现。
- 学生可以开发分布式后门、语义后门以及相应的检测策略，并在隐语平台上实现。

