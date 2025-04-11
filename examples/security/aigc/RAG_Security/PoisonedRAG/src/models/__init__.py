# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




from .PaLM2 import PaLM2
from .Vicuna import Vicuna
from .GPT import GPT
from .Llama import Llama
from .Qwen import Qwen
from .InternLM import InternLM
import json


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        model = PaLM2(config)
    elif provider == 'vicuna':
        model = Vicuna(config)
    elif provider == 'gpt':
        model = GPT(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'qwen':
        model = Qwen(config)
    elif provider == 'internlm':
        model = InternLM(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model
