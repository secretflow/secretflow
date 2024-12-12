# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import transformers

from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.attacks.attack_factory import AttackFactory
from pii_leakage.attacks.extraction.naive_extraction import NaiveExtractionAttack
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.utils.output import print_separator, bcolors, print_dict_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            AttackArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def extract_pii(model_args: ModelArgs,
                ner_args: NERArgs,
                attack_args: AttackArgs,
                env_args: EnvArgs,
                config_args: ConfigArgs):
    """ Generate text using an LM and extract all PII.

    Used for demonstration purposes.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    attack: NaiveExtractionAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    results: dict = attack.attack(lm, verbose=True)

    print_separator()
    print(f"{bcolors.OKBLUE}Best Guess:{bcolors.ENDC} {results}")
    print_separator()

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    extract_pii(*parse_args())
# ----------------------------------------------------------------------------
