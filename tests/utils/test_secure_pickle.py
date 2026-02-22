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


from io import BytesIO

import numpy as np
import pytest

from secretflow.utils import secure_pickle as pickle


def test_should_success_when_load_allowed_class_with_whitelist():
    """test unpickler can load allowed class"""
    obj = np.random.rand(3, 3)
    serialized_obj = pickle.dumps(obj)

    with BytesIO(serialized_obj) as f:
        loaded_obj = pickle.load(f)

    assert isinstance(loaded_obj, np.ndarray)


def test_should_success_when_load_allowed_class_with_blacklist():
    """test unpickler can load allowed class"""
    obj = np.random.rand(3, 3)
    serialized_obj = pickle.dumps(obj)

    with BytesIO(serialized_obj) as f:
        loaded_obj = pickle.load(f, filter_type=pickle.FilterType.BLACKLIST)

    assert isinstance(loaded_obj, np.ndarray)


def test_should_fail_when_modify_allowed_rules():
    with pytest.raises(TypeError):
        pickle._PICKLE_WHITELIST["numpy"] = ["TestClass"]


def test_should_fail_when_add_new_allowed_rules():
    with pytest.raises(TypeError):
        pickle._PICKLE_WHITELIST["tests.utils.test_secure_pickle"] = ["TestClass"]


def test_should_fail_when_load_disallowed_class_with_whitelist():
    class DisAllowedClass:
        pass

    obj = DisAllowedClass()
    serialized_obj = pickle.dumps(obj)

    with BytesIO(serialized_obj) as f:
        with pytest.raises(pickle.UnpicklingError):
            pickle.load(f)


def test_should_fail_when_load_disallowed_class_with_blacklist():
    class EvilObject:
        def __reduce__(self):
            import os

            return (os.system, ("whoami",))

    serialized_obj = pickle.dumps(EvilObject())

    with BytesIO(serialized_obj) as f:
        with pytest.raises(pickle.UnpicklingError):
            pickle.load(f, filter_type=pickle.FilterType.BLACKLIST)
