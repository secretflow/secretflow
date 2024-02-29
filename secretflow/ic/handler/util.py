# Copyright 2023 Ant Group Co., Ltd.
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

import logging
import math
from typing import Optional, Any, Type


def extract_req_algo_params(
    requests: list, algo: int, param_type: Type[Any]
) -> Optional[list]:
    return _extract_enum_params(
        messages=requests,
        enum_field_name='supported_algos',
        param_field_name='algo_params',
        enum_value=algo,
        param_type=param_type,
    )


def extract_req_protocol_family_params(
    requests: list, protocol_family: int, param_type: Type[Any]
) -> Optional[list]:
    return _extract_enum_params(
        messages=requests,
        enum_field_name='protocol_families',
        param_field_name='protocol_family_params',
        enum_value=protocol_family,
        param_type=param_type,
    )


def extract_req_phe_params(
    phe_params: list, algo: int, param_type: Type[Any]
) -> Optional[list]:
    return _extract_enum_params(
        messages=phe_params,
        enum_field_name='supported_phe_algos',
        param_field_name='supported_phe_params',
        enum_value=algo,
        param_type=param_type,
    )


def _extract_enum_params(
    messages: list,
    enum_field_name: str,
    param_field_name: str,
    enum_value: int,
    param_type: Type[Any],
) -> Optional[list]:
    params = []
    try:
        for message in messages:
            enum_field = getattr(message, enum_field_name)
            idx = list(enum_field).index(enum_value)
            param = param_type()
            param_field = getattr(message, param_field_name)
            if not param_field[idx].Unpack(param):
                logging.warning('unpack error')
                return None
            params.append(param)
    except (AttributeError, ValueError):
        logging.exception('extract enum params failed')
        return None
    else:
        return params


def align_param_item(params: list, attr_name: str):
    last_item = None
    for param in params:
        item = getattr(param, attr_name)
        if last_item is None:
            last_item = item
        elif last_item != item:
            return None

    return last_item


def intersect_param_items(params: list, attr_name: str):
    last_items = None
    for param in params:
        item = set(getattr(param, attr_name))
        if last_items is None:
            last_items = item
        else:
            last_items = last_items.intersection(item)

    return last_items


def almost_equal(a, b) -> bool:
    return math.isclose(a, b, rel_tol=1e-6)


def almost_one(a) -> bool:
    return almost_equal(a, 1)
