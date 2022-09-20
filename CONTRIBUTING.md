# Contribution guide

## Build
Please check [INSTALLATION.md](./INSTALLATION.md) to build SecretFlow from source.

## Test

If you need to run all tests,
```sh
./run_pytest.sh
```

If you would like to run a single test, e.g. tests/device/test_spu.py,

```sh
python -m unittest tests/device/test_spu.py
```

## Coding Style
We prefer [black](https://github.com/psf/black) as our code formatter. For various editor users,
please refer to [editor integration](https://black.readthedocs.io/en/stable/integrations/editors.html).
Pass `-S, --skip-string-normalization` to [black](https://github.com/psf/black) to avoid string quotes or prefixes normalization.

