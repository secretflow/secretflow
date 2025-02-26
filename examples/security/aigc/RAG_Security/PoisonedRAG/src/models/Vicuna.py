from fastchat.model import load_model, get_conversation_template
import torch

from .Model import Model


class Vicuna(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.num_gpus = len(config["params"]["gpus"])
        self.max_gpu_memory = config["params"]["max_gpu_memory"]
        self.revision = config["params"]["revision"]
        self.load_8bit = self.__str_to_bool(config["params"]["load_8bit"])
        self.cpu_offloading = self.__str_to_bool(config["params"]["cpu_offloading"])
        self.debug = self.__str_to_bool(config["params"]["debug"])

        self.repetition_penalty = float(config["params"]["repetition_penalty"])

        self.model, self.tokenizer = load_model(
            model_path=self.name,
            device=self.device,
            num_gpus=self.num_gpus,
            max_gpu_memory=self.max_gpu_memory,
            # dtype=torch.float16,
            load_8bit=self.load_8bit,
            cpu_offloading=self.cpu_offloading,
            revision=self.revision,
            debug=self.debug,
        )

    def __str_to_bool(self, s):
        if type(s) == str:
            if s.lower() == 'true':
                return True
            elif s.lower() == 'false':
                return False
        raise ValueError(f'{s} is not a valid boolean')

    def query(self, msg):
        try:
            conv = get_conversation_template(self.name)
            conv.append_message(conv.roles[0], msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer([prompt]).input_ids
            output_ids = self.model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_output_tokens,
            )

            if self.model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]
            outputs = self.tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            response = outputs
        except:
            response = ""

        return response