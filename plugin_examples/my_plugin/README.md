# plugin system

此demo展示了如何扩展一个插件,通常用于业务自定义组件开发。

## 目录结构
``` 
.
├── build.sh
├── my_plugin
│   ├── entry.py            # 插件入口
│   ├── __init__.py
│   └── my_component            
│       ├── __init__.py
│       └── my_component.py # 自定义组件
├── README.md
├── requirements.txt
└── setup.py                # 打包脚本
```

其中setup打包脚本中entry_points入口需要指定group为secretflow_plugins，每个插件名需要全局唯一,entry.py为插件入口函数，需要import所需要的组件

### 手动import

``` python
import os

from .my_component.my_component import MyComponent


_ = MyComponent


# 此main函数不需要做任何事情
def main():
    pass

```

### 自动import,需要注意package的路径是否正确

``` python
import os

from secretflow.component.core import load_component_modules


def main():
    root_path = os.path.dirname(__file__)
    load_component_modules(root_path, "my_plugin", ignore_root_files=True)

```

### 安装

``` shell
# 在dist目录生成whl包,注意:需要预先安装build(pip install build)
python -m build --wheel
# 安装whl包
pip install ./dist/my_plugin-0.1-py3-none-any.whl
```