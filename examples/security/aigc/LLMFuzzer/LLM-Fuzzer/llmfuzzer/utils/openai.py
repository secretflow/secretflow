# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
import logging


def openai_request(messages, model='gpt-3.5-turbo', temperature=1, top_n=1, max_trials=100):
    if openai.api_key is None:
        raise ValueError(
            "You need to set OpenAI API key manually. `opalai.api_key = [your key]`")

    for _ in range(max_trials):
        try:
            results = openai.Completion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=top_n,
            )

            assert len(results['choices']) == top_n
            return results
        except Exception as e:
            logging.warning("OpenAI API call failed. Retrying...")
