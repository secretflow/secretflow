# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from secretflow.device import PYUObject, proxy
from typing import Tuple, Any


@proxy(PYUObject)
class SGBActor:
    def __init__(self):
        self.classes = {}

    def register_class(self, class_name, the_class, *args, **kwargs):
        self.classes[class_name] = the_class(*args, **kwargs)

    def invoke_class_method(self, class_name, function_name, *args, **kwargs):
        return getattr(self.classes[class_name], function_name)(*args, **kwargs)

    def invoke_class_method_two_ret(
        self, class_name, function_name, *args, **kwargs
    ) -> Tuple[Any, Any]:
        return getattr(self.classes[class_name], function_name)(*args, **kwargs)

    def invoke_class_method_three_ret(
        self, class_name, function_name, *args, **kwargs
    ) -> Tuple[Any, Any, Any]:
        return getattr(self.classes[class_name], function_name)(*args, **kwargs)
