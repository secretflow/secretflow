# Contribution guide

## Build
Please check [INSTALLATION.md](./docs/getting_started/installation.md) to build SecretFlow from source.

## Test

Before start, please install required package for testing.
```sh
pip install -r dev-requirements.txt
```

There are two types in secretflow/tests.
- simulation mode: SUT(System under test) is simulated by a single Ray cluster.
- production mode: SUT consists of multiple ray clusters. We have multiple processes running as different parties.
In most cases, SUT is tested by production mode.

If you need to run all tests,

```sh
# tests under production mode
pytest --env prod  -n auto  -v --capture=no  tests/
# tests under simulation mode
pytest --env sim  -n auto  -v --capture=no  tests/
```

If you would like to run a single test, e.g. tests/device/test_spu.py,

```sh
# tests under simulation mode
pytest --env sim  -n auto  -v --capture=no  tests/device/test_spu.py
# tests under production mode
pytest --env prod  -n auto  -v --capture=no  tests/device/test_spu.py
```

## Coding Style
We stick to [Google Style](https://google.github.io/styleguide/pyguide.html).

## Formatter
We prefer [black](https://github.com/psf/black) as our code formatter. For various editor users,
please refer to [editor integration](https://black.readthedocs.io/en/stable/integrations/editors.html).
Pass `-S, --skip-string-normalization` to [black](https://github.com/psf/black) to avoid string quotes or prefixes normalization.

### Imports formatting
`black` won't format your imports. You can do this manually or by IDE, e.g. VSCode. The format of imports refers to [Imports formatting](https://google.github.io/styleguide/pyguide.html#313-imports-formatting)

## Git commit message style
We stick to [Angular Style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines).
