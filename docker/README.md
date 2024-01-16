# Update SecretFlow Meta for Docker Images

```bash
env PYTHONPATH=$PYTHONPATH:$PWD/.. python  update_meta.py
```

1. Skip translation update

```bash
env PYTHONPATH=$PYTHONPATH:$PWD/.. python  update_meta.py -s
```

2. Modify translator, default to 'baidu'. You may check https://pypi.org/project/translators/#supported-translation-services if default value is not available.

```bash
env PYTHONPATH=$PYTHONPATH:$PWD/.. python  update_meta.py -t bing
```
