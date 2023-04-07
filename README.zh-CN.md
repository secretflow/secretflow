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

- [SecretFlow](https://www.secretflow.org.cn/docs/secretflow/zh_CN/)
  - [快速开始](https://www.secretflow.org.cn/docs/secretflow/zh_CN/getting_started/index.html)
  - [组件](https://www.secretflow.org.cn/docs/secretflow/zh_CN/components/index.html)
  - [API文档](https://www.secretflow.org.cn/docs/secretflow/zh_CN/api/index.html)
  - [教程](https://www.secretflow.org.cn/docs/secretflow/zh_CN/tutorial/index.html)

## 安装

请查阅[INSTALLATION.md](./INSTALLATION.md)

## 部署

请查阅[DEPLOYMENT.md](./docs/getting_started/DEPLOYMENT.md)

## 贡献代码

请查阅[CONTRIBUTING.md](./CONTRIBUTING.md)

## 声明

非发布版本的SecretFlow禁止在任何生产环境中使用，因为可能存在漏洞、故障、功能不足、安全问题或其他问题。
