# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum, unique
import io
import pickle
import types

import cloudpickle


# white list of allowed classes for unpickling
# NOTE: use types.MappingProxyType to prevent modification
_PICKLE_WHITELIST = types.MappingProxyType(
    {
        "builtins": ("dict", "list", "tuple", "set", "int", "str", "bytes"),
        "collections": ("defaultdict", "OrderedDict"),
        "scipy.sparse._csr": ("csr_matrix",),
        "numpy": ("ndarray", "dtype"),
        "numpy.core.numeric": ("_frombuffer"),
        "numpy.core.multiarray": ("scalar", "_reconstruct"),
        "pyarrow.lib": ("schema", "field", "type_for_alias"),
        "heu.numpy": ("StringArray", "CiphertextArray"),
        "heu.phe": ("PublicKey",),
        # add more allowed classes here...
    }
)

_PICKLE_BLACKLIST = frozenset(
    [
        'eval',
        'execfile',
        'compile',
        'system',
        'popen',
        'popen2',
        'popen3',
        'popen4',
        'fdopen',
        'tmpfile',
        'fchmod',
        'fchown',
        'openpty',
        'chdir',
        'fchdir',
        'chroot',
        'chmod',
        'chown',
        'lchown',
        'listdir',
        'lstat',
        'mkfifo',
        'mknod',
        'access',
        'mkdir',
        'makedirs',
        'readlink',
        'remove',
        'removedirs',
        'rename',
        'renames',
        'rmdir',
        'tempnam',
        'tmpnam',
        'unlink',
        'execl',
        'execle',
        'execlp',
        'execv',
        'execve',
        'dup2',
        'execvp',
        'execvpe',
        'forkpty',
        'spawnl',
        'spawnle',
        'spawnlp',
        'spawnlpe',
        'spawnv',
        'spawnve',
        'spawnvp',
        'spawnvpe',
        'load',
        'loads',
        'call',
        'check_call',
        'check_output',
        'Popen',
        'getstatusoutput',
        'getoutput',
        'getstatus',
        'getline',
        'copyfileobj',
        'copyfile',
        'copy',
        'copy2',
        'make_archive',
        'listdir',
        'opendir',
        'timeit',
        'repeat',
        'call_tracing',
        'interact',
        'compile_command',
        'spawn',
        'fileopen',
        'getattr',
    ]
)


@unique
class FilterType(Enum):
    """The filter types."""

    WHITELIST = 1
    BLACKLIST = 2


class SecureUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file,
        *,
        fix_imports=True,
        encoding="ASCII",
        errors="strict",
        buffers=None,
        filter_type=FilterType.WHITELIST,
    ):
        super().__init__(
            file,
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        )
        assert isinstance(filter_type, FilterType)
        self.filter_type = filter_type

    def find_class(self, module, name):
        if self.filter_type == FilterType.BLACKLIST:
            # filter by blacklist
            if name.lower() in _PICKLE_BLACKLIST:
                raise pickle.UnpicklingError(
                    f"Attempted to load unauthorized black class: {module}.{name}"
                )
            return super().find_class(module, name)

        # filter by whitelist
        if module.startswith("secretflow."):
            # allow classes from secretflow
            return super().find_class(module, name)

        if module not in _PICKLE_WHITELIST or name not in _PICKLE_WHITELIST[module]:
            raise pickle.UnpicklingError(
                f"Attempted to load unauthorized class: {module}.{name}"
            )

        return super().find_class(module, name)


def load(
    file,
    *,
    fix_imports=True,
    encoding="ASCII",
    errors="strict",
    buffers=None,
    filter_type=FilterType.WHITELIST,
):
    '''
    This function wraps python builtin pickle.load function with whitelist and blacklist.
    You can choose to use whether whitelist or blacklist with parameter filter_type.
    WARNING: use FilterType.BLACKLIST is not perfectly safe as it might be bypassed.
    FilterType.BLACKLIST should be used only when loading pickle target with uncertain type.
    It's preferred to use FilterType.WHITELIST when possible.
    '''
    return SecureUnpickler(
        file,
        fix_imports=fix_imports,
        encoding=encoding,
        errors=errors,
        buffers=buffers,
        filter_type=filter_type,
    ).load()


def loads(
    s,
    /,
    *,
    fix_imports=True,
    encoding="ASCII",
    errors="strict",
    buffers=None,
    filter_type=FilterType.WHITELIST,
):
    '''
    This function wraps python builtin pickle.loads function with whitelist and blacklist.
    You can choose to use whether whitelist or blacklist with parameter filter_type.
    WARNING: use FilterType.BLACKLIST is not perfectly safe as it might be bypassed.
    FilterType.BLACKLIST should be used only when loading pickle target with uncertain type.
    It's preferred to use FilterType.WHITELIST when possible.
    '''
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return load(
        file,
        fix_imports=fix_imports,
        encoding=encoding,
        errors=errors,
        buffers=buffers,
        filter_type=filter_type,
    )


dump, dumps = cloudpickle.dump, cloudpickle.dumps

UnpicklingError = pickle.UnpicklingError
