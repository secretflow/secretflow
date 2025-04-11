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




from openai import OpenAI
from .Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        self.api_keys = config["api_key_info"]["api_keys"]
        # api_pos = int(config["api_key_info"]["api_key_use"])
        # assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        # self.temperature = config["params"]["temperature"]
        self.client = OpenAI(api_key=self.api_keys, base_url="")

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg},
                ],
            )
            # print(completion)
            response = completion.choices[0].message.content

        except Exception as e:
            print(e)
            response = ""

        return response
