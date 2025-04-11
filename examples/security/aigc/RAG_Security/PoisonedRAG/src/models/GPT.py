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
        self.client = OpenAI(api_key=self.api_keys, base_url = "")


    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ]
            )
            # print(completion)
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response