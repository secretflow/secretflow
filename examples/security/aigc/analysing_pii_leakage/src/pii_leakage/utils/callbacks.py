# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import List

from transformers import TrainerCallback, TrainingArguments, TrainerControl, TrainerState

from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from .output import print_highlighted, print_dict_highlighted


class PrintSampleCallback(TrainerCallback):
    """ Generates and prints a single sample using the model.
    """
    def __init__(self, model, sampling_args: SamplingArgs, num_steps: int = 500):
        self.model = model
        self.sampling_args = sampling_args
        self.num_steps = num_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.num_steps == 0:
            sentence = self.model.generate(SamplingArgs(N=1, seq_len=self.sampling_args.seq_len, top_k=self.sampling_args.top_k,
                                                        top_p=self.sampling_args.top_p, generate_verbose=False))
            print_highlighted(sentence)

class EvaluateDPEpsilonCallback(TrainerCallback):
    """ Evaluates the privacy budget of the model
    """
    def __init__(self, model, privacy_accountant, privacy_engine, privacy_args: PrivacyArgs,
                 num_steps: int = 500):
        self.model = model
        self.privacy_accountant = privacy_accountant
        self.privacy_engine = privacy_engine
        self.privacy_args = privacy_args
        self.num_steps = num_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.num_steps == 0:
            eps_prv = self.privacy_accountant.compute_epsilon(self.privacy_engine.steps)[2]
            eps_rdp, alpha = self.privacy_engine.get_privacy_spent(self.privacy_args.target_delta)
            
            eval_data = {
                f"Gstep": state.global_step,
                f"Epoch": state.epoch,
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            }
            
            fp = os.path.join(self.model.get_output_dir(), "training_priv.json")
            #data[state.global_step] = eval_data
            print_dict_highlighted(eval_data)


class EvaluatePerplexityCallback(TrainerCallback):
    def __init__(self, dataset: List, model, prefix: str = "PPL", num_steps: int = 500, verbose=True):
        self.dataset = dataset
        self.prefix = prefix
        self.num_steps = num_steps
        self.verbose = verbose
        self.model = model

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.num_steps is not None and state.global_step % self.num_steps == 0:

            ppl = self.model.perplexity(self.dataset["text"])
            print_highlighted(f"{self.prefix}={ppl}")
