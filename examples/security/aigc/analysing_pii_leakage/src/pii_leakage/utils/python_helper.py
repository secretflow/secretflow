from enum import Enum

import hashlib


import numpy as np
import json

import torch


def deduplicate_list_of_dicts(list_of_dicts):
    unique_dicts = []
    for dict_item in list_of_dicts:
        if dict_item not in unique_dicts:
            unique_dicts.append(dict_item)
    return unique_dicts

class DynamicEnum(str, Enum):
    def __str__(self):
        return self.name

def hash_dict(d):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.cpu().numpy().tolist()
        raise TypeError(f"Not serializable: {type(o)}")

    sorted_dict = {k: v for k, v in sorted(d.items())}
    json_str = json.dumps(sorted_dict, default=default, sort_keys=True)
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()