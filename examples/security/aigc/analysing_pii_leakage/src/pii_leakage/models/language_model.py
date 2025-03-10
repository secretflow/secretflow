# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict

from accelerate import Accelerator

import dp_transformers
import numpy as np
import torch
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    TrainerCallback, BitsAndBytesConfig

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset
from ..utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback
from ..utils.output import print_highlighted
from ..utils.web import is_valid_url, download_and_unzip
from peft import LoraConfig, PeftConfig

from trl import SFTTrainer

# from alignment import (
#     get_kbit_device_map,
#     get_peft_config,
#     get_quantization_config,
# )

def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None

def get_peft_config(model_args) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config

def get_quantization_config(model_args) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        ).to_dict()
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        ).to_dict()
    else:
        quantization_config = None

    return quantization_config

@dataclass
class GeneratedText:
    text: str  # the generated text
    #score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])


class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """ A wrapper class around a huggingface LM.
        """
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """ Gets the maximum size of the context """
        if "gpt" in self.model_args.architecture:
            return self._lm.config.n_positions
        else:
            return self._lm.config.max_position_embeddings

    @abstractmethod
    def tokenizer(self):
        """ Returns this model's tokenizer. """
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def load(self, verbose: bool = False) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint.
        """
        #######################
        # Load pretrained model
        #######################
        print("*** Load pretrained model ***")
        torch_dtype = (
            self.model_args.torch_dtype if self.model_args.torch_dtype in ["auto", None] else getattr(torch, self.model_args.torch_dtype)
        )
        quantization_config = get_quantization_config(self.model_args)

        model_kwargs = dict(
            revision=self.model_args.model_revision,
            trust_remote_code=self.model_args.trust_remote_code,
            attn_implementation=self.model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'.")
                
            self._lm = AutoModelForCausalLM.from_pretrained(self.model_args.model_ckpt, **model_kwargs)
        elif self.model_args.pre_trained:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")
            self._lm = AutoModelForCausalLM.from_pretrained(self.model_args.architecture, **model_kwargs)
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = AutoModelForCausalLM(config=self.get_config())
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_args.architecture)
        self._tokenizer.padding_side = 'right'
        
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._lm.resize_token_embeddings(len(self._tokenizer))

        self._lm.to(self.env_args.device)
        return self

    def substring_perplexity(self, seq: str, substring: str) -> float:
        """ Computes the perplexity of a substring in a string.
        For example: seq="My name is Ronald and I like hamburgers.", substring="Ronald",
        then this function computes the perplexity of generating "Ronald" given prefix "My name is".
        """
        original_mode = self._lm.training
        self._lm.eval()

        txt = seq[:seq.index(substring) + len(substring)]
        input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
        substring_len = len(self._tokenizer.encode(substring, truncation=True))
        target_ids = input_ids.clone()
        target_ids[:, :input_ids.size(1) - substring_len] = -100
        with torch.no_grad():
            outputs = self._lm(input_ids, labels=target_ids)
        loss, _, num_tokens = outputs[:3]

        perplexity = torch.exp(loss / num_tokens)

        self._lm.training = original_mode
        return perplexity.cpu().item()

    def autocomplete(self, sampling_args: SamplingArgs):
        """ Predicts the top-1 most probable next tokens. """
        return self.generate(sampling_args)[0]

    def print_sample(self, prompt=None):
        self._lm.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64))
        print_highlighted(data[0].text)
        return data[0].text

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, sampling_args) -> List[GeneratedText]:
        """ Helper function to generate a single batch of text.
        """
        self._lm.eval()

        input_len = input_ids.size(1)
        out = self._lm.generate(
            input_ids=input_ids.to(self.env_args.device),
            attention_mask=attention_mask.to(self.env_args.device),
            max_length=min(self.n_positions, input_len + sampling_args.seq_len),
            # max_new_tokens=sampling_args.seq_len,
            max_length=input_len + sampling_args.seq_len,
            do_sample=sampling_args.do_sample,
            top_k=sampling_args.top_k,
            top_p=sampling_args.top_p,
            output_scores=False,
            return_dict_in_generate=True
        )

        generated_texts: List[GeneratedText] = []
        for text in self._tokenizer.batch_decode(out.sequences, skip_special_tokens=False):
            generated_texts.append(GeneratedText(text=text))
        return generated_texts


    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using the sampling args.
        """
        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] * r if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt] * r
        )

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        generated_data: List[GeneratedText] = []
        num_batches = int(np.ceil(sampling_args.N / self.env_args.eval_batch_size))
        for _ in tqdm(
                range(num_batches),
                disable=not sampling_args.generate_verbose,
                desc="Generating with LM"
        ):
            generated_data.extend(self.generate_batch(input_ids, attention_mask, sampling_args))

        return GeneratedTextList(data=generated_data)

    def tokenize_datasets(self, datasets: List[RealDataset], column_name="text") -> List:
        """ Tokenizes the 'text' column of a list of dataset using this model's tokenizer """
        tokenize_function = lambda x: self._tokenizer(x[column_name], truncation=True)
        return [dataset.get_hf_dataset().map(tokenize_function, batched=True).select_columns(['input_ids', 'attention_mask']) for dataset in datasets]

    def perplexity(self, data: Union[list, str], offset=0, max_length=0, apply_exp=True, verbose=True,
                   return_as_list: bool = False) -> float:
        """ Compute the perplexity of the model on a string.
        """
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of tokens viewed
        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
            input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
            target_ids = input_ids.clone()

            if offset > 0:  # ignore everything up to the offset
                target_ids[:, :offset] = -100

            tgt_len = (target_ids.size(1) - offset)
            if max_length > 0:  # ignore everything except offset:offset+max_length
                target_ids[:, offset + max_length:] = -100
                tgt_len = max_length

            with torch.no_grad():
                outputs = self._lm(input_ids, labels=target_ids)
            loss, logits = outputs[:2]
            if return_as_list:
                nlls.append(loss.cpu().detach())
            else:
                nlls.append(loss.cpu().detach())
                ctr += tgt_len

        self._lm.training = original_mode
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())

    def fine_tune_dp(self,
                      train_dataset: RealDataset,
                      eval_dataset: RealDataset,
                      train_args: TrainerArgs,
                      privacy_args: PrivacyArgs):

        with train_args.main_process_first(desc="Tokenizing datasets"):
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])

        self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
                                                            target_epsilon=privacy_args.target_epsilon,
                                                            target_delta=privacy_args.target_delta,
                                                            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
            tokenizer=self._tokenizer
        )

        # Workaround for modern `transformers` which removed `use_cuda_amp` 
        # (See https://github.com/huggingface/transformers/pull/25702)
        trainer.use_cuda_amp = False

        try:
            trainer.train()
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })

        trainer.save_model()
        self._lm.eval()
