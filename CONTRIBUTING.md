# Contribution guide

## Build
Please check [INSTALLATION.md](./docs/getting_started/installation.md) to build SecretFlow from source.

## Test

Before start, please install required package for testing.
```sh
pip install -r requirements.txt -r dev-requirements.txt
```

There are two types in secretflow/tests.
- simulation mode: SUT(System under test) is simulated by a single Ray cluster.
- production mode: SUT consists of multiple ray clusters. We have multiple processes running as different parties.
In most cases, SUT is tested by production mode.

If you need to run all tests,

```sh
# tests under production mode
pytest --env prod -v --capture=no  tests/
# tests under simulation mode
pytest --env sim -v --capture=no  tests/
```

If you would like to run a single test, e.g. tests/device/test_spu.py,

```sh
# tests under simulation mode
pytest --env sim -v --capture=no  tests/device/test_spu.py
# tests under production mode
pytest --env prod -v --capture=no  tests/device/test_spu.py
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


## Documentation update and its multilingual version
if you update the documentation files in `docs/`, you are supposed to update its multilingual version in `docs/locales/zh_CN/LC_MESSAGES`.

For example, if you update `docs/getting_started/installation.md`, you also need to update the corresponding `*.po` file in `docs/locales/zh_CN/LC_MESSAGES/getting_started/installation.po`.

Follow follwing steps to update documentation:
1. Update documentation in `docs`.
2. Run the following command to update `*.po` file.

   ```bash
   cd docs
   pip install -r requirements.txt
   sh update_po.sh
   ```
3. Update the corresponding `*.po` file.
4. All `fuzzy` should be removed in `*.po` file, because it won't take effect in the Chinese version of the documentation.
5. All strings which start with `#~` such as `#~ msgid ` or `#~ msgstr` should be removed, because it is redundant.
6. Only commit the files which you update and pull request.
7. If your document is conflict with the main branch of SecretFlow, you are supposed to solve the conflict locally and commit.


## Compiling Protocol Buffers

You should use [protoc v3.19.6](https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.6)


### Compiling SecretFlow Open Specification

Protocol Buffers resides at submodules/spec/ as git submodules.

```bash

~/protoc-3.19.6/bin/protoc --proto_path submodules/spec/ --python_out . submodules/spec/secretflow/spec/v1/*.proto

```

### Compiling Extended Specification

Protocol Buffers resides at secretflow/protos.

```bash
~/protoc-3.19.6/bin/protoc --proto_path secretflow/protos/ --python_out . secretflow/protos/secretflow/spec/extend/*.proto
```

All generated Python code resides at secretflow/spec.

