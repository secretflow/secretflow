# Installation
## Environment
Python：3.8

OS: CentOS 7, Ubuntu 18.04

CPU/Memory: recommended minimum requirement is 8C16G.

## Installation

The latest release package can be obtained from [pypi](https://pypi.org).

```bash
pip install -U secretflow
```
## Have a try
After installation, you can try running your secretflow code.

```python
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=True)
>>> dev = sf.PYU('alice')
>>> import numpy as np
>>> data = dev(np.random.rand)(3, 4)
>>> data
<secretflow.device.device.pyu.PYUObject object at 0x7fdec24a15b0>
```