<div align="center">
    <img src="docs/_static/logo-light.png">
</div>

---

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/secretflow/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/secretflow/tree/main)

<p align="center">
<a href="./README.zh-CN.md">简体中文</a>｜<a href="./README.md">English</a>
</p>

SecretFlow是一个统一的框架，用于保护隐私的数据智能和机器学习。为了实现这个目标，它提供了以下内容：
- 抽象设备层，包括明文设备和封装了各种密态协议的密态设备。
- 设备流层，将高阶算法转为设备对象流和DAG。
- 算法层，使用水平或垂直分区的数据进行数据分析和机器学习。
- 工作流层，无缝集成数据处理、模型训练和超参调整。

<div align="center">
    <img src="docs/_static/secretflow_arch.svg">
</div>

## 文档

- [SecretFlow](https://www.secretflow.org.cn/docs/secretflow/)
  - [快速开始](https://www.secretflow.org.cn/docs/secretflow/getting_started)
  - [用户指南](https://www.secretflow.org.cn/docs/secretflow/user_guide)
  - [API文档](https://www.secretflow.org.cn/docs/secretflow/api)
  - [教程](https://www.secretflow.org.cn/docs/secretflow/tutorial)

## 相关项目
- [Kuscia](https://github.com/secretflow/kuscia): 一款基于 K3s 的轻量级隐私计算任务编排框架。
- [SCQL](https://github.com/secretflow/scql): 允许多个不信任方在不泄露其私人数据的情况下进行联合分析的系统。
- [SPU](https://github.com/secretflow/spu): 一种可证明、可测量的安全计算设备，在提供计算能力的同时保护您的数据隐私。
- [HEU](https://github.com/secretflow/heu): 一个高性能的同态加密算法库。
- [YACL](https://github.com/secretflow/yacl): 一个 C++ 库，包含SecretFlow代码所依赖的密码学、网络和io模块。

## 安装

请查阅[INSTALLATION.md](./docs/getting_started/installation.md)

## 部署

请查阅[DEPLOYMENT.md](./docs/getting_started/deployment.md)

## 学习隐私计算技术

我们同样提供了关于隐私计算技术学术论文的总结，以及某些论文的解读。

请查阅 [AWESOME-PETS.md](./docs/awesome-pets/awesome-pets.md)

## 贡献代码

请查阅[CONTRIBUTING.md](./CONTRIBUTING.md)

## 声明

非发布版本的SecretFlow禁止在任何生产环境中使用，因为可能存在漏洞、故障、功能不足、安全问题或其他问题。
