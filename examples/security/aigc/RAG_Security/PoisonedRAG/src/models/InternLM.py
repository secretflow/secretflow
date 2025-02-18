import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from .Model import Model


class InternLM(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        # self.model_path = "./model/llama2_7b"

        # api_pos = int(config["api_key_info"]["api_key_use"])
        # hf_token = config["api_key_info"]["api_keys"][api_pos]
        # tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
        # 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
        # model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", device_map="auto",trust_remote_code=True, torch_dtype=torch.float16)
        # (可选) 如果在低资源设备上，可以通过bitsandbytes加载4-bit或8-bit量化的模型，进一步节省GPU显存.
        # 4-bit 量化的 InternLM 7B 大约会消耗 8GB 显存.
        # pip install -U bitsandbytes
        # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
        # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(self.name,  device_map="auto", trust_remote_code=True, load_in_8bit=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.name, device_map="auto", trust_remote_code=True, load_in_4bit=True).to(self.device)

    def query(self, msg):
        # input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to("cuda")
        # outputs = self.model.generate(input_ids,
        #     temperature=self.temperature,
        #     max_new_tokens=self.max_output_tokens,
        #     early_stopping=True)
        # out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # result = out[len(msg):]
        response, history = self.model.chat(self.tokenizer, msg, history=[])
        return response