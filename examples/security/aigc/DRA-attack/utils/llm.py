import time
import anthropic
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:

    def generate(self, prompt):
        raise NotImplementedError


class GPT(LLM):
    
    def __init__(self, args):
        super().__init__()
        self.client = OpenAI(
            api_key=args.api_key_gpt,
            base_url=args.base_url_gpt,
        )
        self.model = args.model
        

    def generate(
        self,
        prompt,
    ):
        temperature = 0.0
        n = 1
        max_trial = 50
        for _ in range(max_trial):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    n=n,
                    max_tokens=256,
                )
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                continue

        return response.choices[0].message.content


class Claude(LLM):

    def __init__(self, args):
        super().__init__()
        self.client = anthropic.Anthropic(
            api_key= args.api_key_claude,
            base_url=args.base_url_claude
        )
        self.model = args.model
    
    def generate(
        self,
        prompt,
    ):
        max_trial = 50
        for _ in range(max_trial):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[
                        {"role": "user", "content": f"{prompt}"}
                    ]
                )
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                continue

        return response.content[0].text


class Qwen(LLM):

    def __init__(self, args):
        super().__init__()
        self.client = OpenAI(
            api_key=args.api_key_qwen,
            base_url=args.base_url_qwen,
        )
        self.model = args.model
    
    def generate(
        self,
        prompt,
    ):
        temperature = 0.0
        n = 1
        max_trial = 50
        for _ in range(max_trial):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    n=n,
                    max_tokens=256,
                )
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                continue

        return response.choices[0].message.content

class LocalLLM(LLM):

    def __init__(self, args):
        super().__init__()
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, local_files_only=args.use_local_model
            )
            .to(args.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, local_files_only=args.use_local_model
        )
        self.template = args.template
        self.device = args.device
    
    def generate(self, prompt):
        conv_prompt = self.template.format(instruction=prompt)
        input_ids = self.tokenizer(conv_prompt, return_tensors="pt").to(self.device)["input_ids"]
        output_ids = self.model.generate(input_ids, max_new_tokens=800)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(conv_prompt) :]
        return response
    