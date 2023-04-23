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

We also provide a curated list of papers and SecretFlow's tutorials on Privacy-Enhancing Technologies (PETs).

Please check [AWESOME-PETS.md](./docs/awesome-pets/README.md)


## Contributing

Please check [CONTRIBUTING.md](./CONTRIBUTING.md)

## Disclaimer

Non-release versions of SecretFlow are prohibited to use in any production environment due to possible bugs, glitches, lack of functionality, security issues or other problems.
