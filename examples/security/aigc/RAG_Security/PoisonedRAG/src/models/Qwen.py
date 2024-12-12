import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from .Model import Model


class Qwen(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        # self.model_path = "./model/llama2_7b"

        # api_pos = int(config["api_key_info"]["api_key_use"])
        # hf_token = config["api_key_info"]["api_keys"][api_pos]

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        # self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(self.name).to(self.device)
        

    def query(self, msg):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": msg}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_output_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


        # outputs = self.model.generate(input_ids,
        #     temperature=self.temperature,
        #     max_new_tokens=self.max_output_tokens,
        #     early_stopping=True)
        # out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # result = out[len(msg):]
        return result