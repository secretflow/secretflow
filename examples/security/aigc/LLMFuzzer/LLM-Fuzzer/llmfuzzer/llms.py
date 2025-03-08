import time
import anthropic
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LLAMA_SYSTEM_MESSAGE = (
    "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
    "\n\n{instruction} [/INST]"
)

DEFAULT_VICUNA_SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    "USER: {instruction}\nASSISTANT: "
)

DEFAULT_GPT_SYSTEM_MESSAGE = "You are a helpful assistant."



class LLM:

    def generate(self, prompt):
        raise NotImplementedError


class GPT(LLM):

    def __init__(
        self,
        model_name,
        api_key,
        logger,
        base_url=None,
        system_message=None,
    ):
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.logger = logger
        self.model_name = model_name
        self.system_message = (
            DEFAULT_GPT_SYSTEM_MESSAGE if not system_message else system_message
        )

    def generate(
        self,
        prompt,
        temperature=0,
        max_tokens=512,
        max_trials=10,
        failure_sleep_time=2,
    ):
        for _ in range(max_trials):
            pass
            try:
                results = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return results.choices[0].message.content
            except Exception as e:
                self.logger.error(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times..."
                )
                time.sleep(failure_sleep_time)

        return [" "]


class Claude(LLM):

    def __init__(
        self,
        model_name,
        api_key,
        logger,
        base_url=None,
    ):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.logger = logger
        self.model_name = model_name

    def generate(
        self,
        prompt,
        temperature=0,
        max_tokens=512,
        max_trials=10,
        failure_sleep_time=2,
    ):
        for _ in range(max_trials):
            pass
            try:

                results = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": f"{prompt}"}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return results.content[0].text
            except Exception as e:
                self.logger.error(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times..."
                )
                time.sleep(failure_sleep_time)

        return [" "]


class Qwen(LLM):

    def __init__(
        self,
        model_name,
        api_key,
        logger,
        base_url=None,
    ):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.logger = logger
        self.model_name = model_name

    def generate(
        self,
        prompt,
        temperature=0,
        max_tokens=512,
        max_trials=1,
        failure_sleep_time=2,
    ):
        for _ in range(max_trials):
            pass
            try:

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': str(prompt)}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
    
                return completion.choices[0].message.content

            except Exception as e:
                self.logger.error(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times..."
                )
                time.sleep(failure_sleep_time)

        return [" "]


class LocalLLM(LLM):
    def __init__(self, model_name_or_path, device, logger, system_message=None):
        super().__init__()
        self.device = device
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        logger.info(f"Model: {model_name_or_path} loaded.")

        if "llama" in model_name_or_path.lower():
            self.system_message = (
                DEFAULT_LLAMA_SYSTEM_MESSAGE if not system_message else system_message
            )
        elif "vicuna" in model_name_or_path.lower():
            self.system_message = (
                DEFAULT_VICUNA_SYSTEM_MESSAGE if not system_message else system_message
            )
        else:
            raise NotImplementedError
            

    def generate(
        self,
        prompt,
        max_tokens=512,
    ):
        conv_prompt = self.system_message.format(instruction=prompt)
        input_ids = self.tokenizer(conv_prompt, return_tensors="pt").to(self.device)[
            "input_ids"
        ]
        output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)[
            len(conv_prompt) :
        ]
        return response

