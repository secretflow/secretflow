# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pprint import pprint

from accelerate import Accelerator
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, Trainer
from trl import SFTTrainer
from peft import LoraConfig, PeftConfig
import torch

from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.outdir_args import OutdirArgs
from pii_leakage.arguments.privacy_args import PrivacyArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.arguments.trainer_args import TrainerArgs
from pii_leakage.dataset.real_dataset import RealDataset
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted
from pii_leakage.utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback


def parse_args():
    parser = HfArgumentParser((ModelArgs,
                                NERArgs,
                                TrainerArgs,
                                DatasetArgs,
                                PrivacyArgs,
                                OutdirArgs,
                                EnvArgs,
                                ConfigArgs))
    return parser.parse_args_into_dataclasses()

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

def fine_tune(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              privacy_args: PrivacyArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """ Fine-tunes a language model (LM) on some text dataset with/without privacy.
    """
    accelerator = Accelerator()  # Initialize Accelerator

    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        privacy_args = config_args.get_privacy_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(config_args.get_privacy_args()))

    # -- Load the datasets
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("validation"),
                                                                 ner_args=ner_args, env_args=env_args)

    # -- Load the LM
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    # -- Print configuration
    output_folder = outdir_args.create_folder_name()

    print_highlighted(f"Saving LM to: {output_folder}. Train Size: {len(train_dataset)},"
                      f" Eval Size: {len(eval_dataset)}")
    print_highlighted(f"Train Sample: {train_dataset.shuffle().first()}")

    if privacy_args.target_epsilon > 0: # TODO: use dp 
        lm.fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
    else:
        extra_callbacks = []
        extra_callbacks += [
            PrintSampleCallback(
                model=lm,
                sampling_args=SamplingArgs(),
                num_steps=train_args.callback_after_n_steps
            ),
            EvaluatePerplexityCallback(
                dataset=eval_dataset,
                model=lm,
                prefix="Eval PPL",
                num_steps=train_args.callback_after_n_steps
            )
        ]

        # Prepare data
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=lm._tokenizer,
            mlm=False
        )

        print("Tokenizing Train and Eval Datasets ...")
        eval_dataset = eval_dataset.shuffle().select(
            list(range(train_args.limit_eval_dataset))
        )
        train_dataset, eval_dataset = lm.tokenize_datasets([train_dataset, eval_dataset])
        print("Done Tokenizing!")

        # Wrap model, optimizer, and datasets with Accelerator
        train_dataset, eval_dataset = accelerator.prepare(
            train_dataset, eval_dataset
        )

        # Configure training
        train_args.eval_strategy = "no"

        ########################
        # Initialize the Trainer
        ########################
        if model_args.use_peft:
            trainer = SFTTrainer(
                model=lm._lm,
                args=train_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=lm._tokenizer,
                packing=False,
                peft_config=get_peft_config(lm.model_args),
                data_collator=data_collator,
                callbacks=extra_callbacks
            )
            trainer.model.print_trainable_parameters()
        else:
            trainer = Trainer(model=lm._lm,
                args=train_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=extra_callbacks)

        ###############
        # Training loop
        ###############
        print("*** Train ***")

        train_result = trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        ##################################
        # Save model and create model card
        ##################################
        print("*** Save model ***")
        trainer.save_model()
    
        lm._lm.eval()

    # -- Print using the LM
    pprint(lm.generate(SamplingArgs(N=1)))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fine_tune(*parse_args())
# ----------------------------------------------------------------------------
