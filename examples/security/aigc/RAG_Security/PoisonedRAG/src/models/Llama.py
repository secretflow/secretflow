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




import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from .Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        # self.model_path = "./model/llama2_7b"

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        self.tokenizer = LlamaTokenizer.from_pretrained(self.name)
        self.model = LlamaForCausalLM.from_pretrained(
            self.name, torch_dtype=torch.float16
        ).to(self.device)

    def query(self, msg):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(
            input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True,
        )
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(msg) :]
        return result
