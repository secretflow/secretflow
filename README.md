<div align="center">
    <img src="docs/_static/logo-light.png">
</div>

---

SecretFlow is a unified framework for privacy-preserving data intelligence and machine learning. To achieve this goal,
it provides:

- An abstract device layer consists of plain devices and secret devices which encapsulate various cryptographic protocols.
- A device flow layer modeling higher algorithms as device object flow and DAG.
- An algorithm layer to do data analysis and machine learning with horizontal or vertical partitioned data.
- A workflow layer that seamlessly integrates data processing, model training, and hyperparameter tuning.

<div align="center">
    <img src="docs/_static/secretflow_arch.svg">
</div>

## Install

For users who want to try SecretFlow, you can install the current release
from [pypi](https://pypi.org/). Note that it requires python version == 3.8,
you can create a virtual environment with conda if not satisfied.

```sh
pip install -U secretflow
```

Try you first SecretFlow program

```python
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=True)
>>> dev = sf.PYU('alice')
>>> import numpy as np
>>> data = dev(np.random.rand)(3, 4)
>>> data
<secretflow.device.device.pyu.PYUObject object at 0x7fdec24a15b0>
```

## Getting started

- [Getting started](https://secretflow.readthedocs.io/en/latest/getting_started/index.html)
- [Tutorials help you to understand and use secretflow](https://secretflow.readthedocs.io/en/latest/tutorial/index.html)
- [The api reference](https://secretflow.readthedocs.io/en/latest/reference/index.html)

## Deployment

- [Standalone or Cluster Mode](docs/getting_started/deployment.md)

## Contribution guide

For developers who want to contribute to SecretFlow, you can set up an environment with the following instruction.

```sh
git clone https://github.com/secretflow/secretflow.git

# optional
git lfs install

conda create -n secretflow python=3.8
conda activate secretflow
pip install -r dev-requirements.txt -r requirements.txt
```

### Coding Style
We prefer [black](https://github.com/psf/black) as our code formatter. For various editor users,
please refer to [editor integration](https://black.readthedocs.io/en/stable/integrations/editors.html).
Pass `-S, --skip-string-normalization` to [black](https://github.com/psf/black) to avoid string quotes or prefixes normalization.

## Disclaimer
Non-release versions of SecretFlow are prohibited to use in any production environment due to possible bugs, glitches, lack of functionality, security issues or other problems.
