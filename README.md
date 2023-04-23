<div align="center">
    <img src="docs/_static/logo-light.png">
</div>

---

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/secretflow/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/secretflow/tree/main)

<p align="center">
<a href="./README.zh-CN.md">简体中文</a>｜<a href="./README.md">English</a>
</p>

SecretFlow is a unified framework for privacy-preserving data intelligence and machine learning. To achieve this goal,
it provides:

- An abstract device layer consists of plain devices and secret devices which encapsulate various cryptographic protocols.
- A device flow layer modeling higher algorithms as device object flow and DAG.
- An algorithm layer to do data analysis and machine learning with horizontal or vertical partitioned data.
- A workflow layer that seamlessly integrates data processing, model training, and hyperparameter tuning.

<div align="center">
    <img src="docs/_static/secretflow_arch.svg">
</div>

## Documentation

- [SecretFlow](https://www.secretflow.org.cn/docs/secretflow/en/)
  - [Getting Started](https://www.secretflow.org.cn/docs/secretflow/en/getting_started/index.html)
  - [Components](https://www.secretflow.org.cn/docs/secretflow/en/components/index.html)
  - [API Reference](https://www.secretflow.org.cn/docs/secretflow/en/api/index.html)
  - [Tutorial](https://www.secretflow.org.cn/docs/secretflow/en/tutorial/index.html)


## SecretFlow Related Projects

- [SCQL](https://github.com/secretflow/scql): A system that allows multiple distrusting parties to run joint analysis without revealing their private data.
- [SPU](https://github.com/secretflow/spu): A provable, measurable secure computation device, which provides computation ability while keeping your private data protected.
- [HEU](https://github.com/secretflow/heu): A high-performance homomorphic encryption algorithm library.
- [YACL](https://github.com/secretflow/yacl): A C++ library that contains cryptgraphy, network and io modules which other SecretFlow code depends on.

## Install

Please check [INSTALLATION.md](./docs/getting_started/installation.md)

## Deployment

Please check [DEPLOYMENT.md](./docs/getting_started/deployment.md)

## Learn PETs

We also provide a curated list of papers and SecretFlow's tutorials on Privacy-Enhancing Technologies (PETs). In this list, we only define and categorie techniques that help maintain the security and privacy of data.

Please feel free to open a pull request.

Cryptography-based techniques. If you are a beginner in cryptography, and wants to learn about the theory or applications of crypto, please also checkout our [[crypto-for-beginners]](docs/tutorialscrypto-beginner.md) list.

1. [Secure Multi-Party Computation (MPC)](docs/paperstools/mpc.md)  (Contributors: [@jamie-cui](https://www.github.com/jamie-cui))
2. [Zero-Knowledge Proof (ZKP)](docs/paperstools/zkp.md) (Contributors: [@xfap](https://www.github.com/xfap))

Anonymity-related techniques

1. [Differential Privacy (DP)](docs/paperstools/dp.md) (Contributors: [@yingting6](https://www.github.com/yingting6))

Hardware-based solutions

1. [Trusted Execution Environment (TEE)](docs/paperstools/tee.md) 

Private Set/Database Operations

1. [Private Set Intersection (PSI)](docs/papersapplications/set/psi.md) (Contributors: [@jamie-cui](https://www.github.com/jamie-cui))

Protecting training/inference data

1. [PPML based on Crypto](docs/papersapplications/ppml/ppml_crypto.md) (Contributors: [@llCurious](https://www.github.com/llCurious))
2. [Ferderated Learning (FL)](docs/papersapplications/ppml/fl/fl.md) (Contributors: [@zhangxingmeng](https://www.github.com/zhangxingmeng) [@FelixZheng1](https://www.github.com/FelixZheng1))

Attacks on machine learning system

1. [General attacks and defense](docs/papersapplications/aml/attack_defense.md) (Contributors: [@zhangxingmeng](https://www.github.com/zhangxingmeng))

Multimedia Privacy and Security

1. [Summaries and Talks](docs/papersapplications/multimedia/summary.md) (Contributors: [@XiaoHwei](https://www.github.com/XiaoHwei))
2. [Attack Methods](docs/papersapplications/multimedia/attack.md) (Contributors: [@XiaoHwei](https://www.github.com/XiaoHwei))
3. [Defense Methods](docs/papersapplications/multimedia/defense.md) (Contributors: [@XiaoHwei](https://www.github.com/XiaoHwei))


We're sorry some of the materials are avaliable only in Chinese, we'll try to provide an English version in the future.

1. [Team SecretFlow's Papers](docs/paperssecretflow.md)
2. [Team SecretFlow's Talks (on bilibili.com)](docs/tutorialsbilibili.md) 
3. [Team SecretFlow's Posts (on wechat)](docs/tutorialswechat.md)

## Contributing

Please check [CONTRIBUTING.md](./CONTRIBUTING.md)

## Disclaimer

Non-release versions of SecretFlow are prohibited to use in any production environment due to possible bugs, glitches, lack of functionality, security issues or other problems.
