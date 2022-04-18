# 安装

> 💡 注意，您的python版本需要>=3.8。推荐使用conda管理您的python环境。

```bash
pip install secretflow
```

安装后，可以尝试运行你的第一份隐语代码。

```python
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=True)
>>> dev = sf.PYU('alice')
>>> import numpy as np
>>> data = dev(np.random.rand)(3, 4)
>>> data
<secretflow.device.device.pyu.PYUObject object at 0x7fdec24a15b0>
```