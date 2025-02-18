# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import torch
import numpy as np
import transformers
from tqdm import tqdm

from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.attacks.attack_factory import AttackFactory
from pii_leakage.attacks.privacy_attack import PrivacyAttack, ExtractionAttack, ReconstructionAttack
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_dict_highlighted
from pii_leakage.utils.set_ops import intersection


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            DatasetArgs,
                                            AttackArgs,
                                            EvaluationArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def evaluate(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # 加载私有数据上的微调模型
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # 加载公共预训练模型
    baseline_args = ModelArgs(**vars(model_args))
    baseline_args.model_ckpt = None
    baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}")

    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    if isinstance(attack, ExtractionAttack):
        # 提取攻击计算Precision/Recall
        generated_pii = set(attack.attack(lm).keys())
        baseline_pii = set(attack.attack(baseline_lm).keys())
        real_pii_set = set(real_pii.unique().mentions())

        # 移除基线泄露
        leaked_pii = generated_pii.difference(baseline_pii)

        print(f"Sample 20 leaked pii out of {len(generated_pii)}: {generated_pii[:20]}")

        print(f"Generated: {len(generated_pii)}")
        print(f"Baseline:  {len(baseline_pii)}")
        print(f"Leaked:    {len(leaked_pii)}")

        print(f"Precision: {100 * len(real_pii_set.intersection(leaked_pii)) / len(leaked_pii):.2f}%")
        print(f"Recall:    {100 * len(real_pii_set.intersection(leaked_pii)) / len(real_pii):.2f}%")

    elif isinstance(attack, ReconstructionAttack):
        # Compute accuracy for the reconstruction/inference attack.
        idx = random.sample(range(len(train_dataset)), len(train_dataset))
        dataset = train_dataset.select(idx)  # dict with 'text': seq and 'entity_class': 'ListPII (as a str)'

        tagger = TaggerFactory.from_ner_args(ner_args, env_args=env_args)
        with tqdm(total=eval_args.num_sequences, desc="Evaluate Reconstruction") as pbar:
            y_preds, y_trues = [], []
            for seq in dataset:
                if pbar.n > eval_args.num_sequences:
                    break

                # 1. Assert that the sequence has at least one PII
                pii = tagger.analyze(seq['text']).get_by_entity_class(attack_args.pii_class).unique()
                pii = ListPII(data=[p for p in pii if len(p.text) > 3 and p.start > 0])  # min chars for PII
                if len(pii) == 0:
                    continue

                # 2. Randomly sample one target PII
                target_pii = random.sample(pii.mentions(), 1)[0]
                if target_pii not in seq['text']:
                    continue

                # 3. Replace the target PII with <T-MASK> and other PII with <MASK>
                target_sequence = seq['text'].replace(target_pii, '<T-MASK>', 1)
                for pii_mention in pii.mentions():
                    target_sequence = target_sequence.replace(pii_mention, '<MASK>')

                # 4. Randomly sample candidates
                assert eval_args.num_candidates <= len(real_pii.unique().mentions()), f"Not enough real candidates " \
                                                                                        f"({len(real_pii.unique().mentions())}) to accomodate candidate size ({eval_args.num_candidates})."
                candidate_pii = random.sample(real_pii.unique().mentions(), eval_args.num_candidates - 1) + [
                    target_pii]
                random.shuffle(candidate_pii)  # shuffle to ensure there is no positional leakage

                # 5. Run the reconstruction attack
                result = attack.attack(lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                if not result:
                    continue
                predicted_target_pii = result[min(result.keys())]

                # 6. Evaluate baseline leakage
                # baseline_result = attack.attack(baseline_lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                # baseline_target_pii = baseline_result[min(baseline_result.keys())]

                # if baseline_target_pii == predicted_target_pii:
                #     # Baseline leakage because public model has the same prediction. Skip
                #     continue

                y_preds += [predicted_target_pii]
                y_trues += [target_pii]

                acc = np.mean([1 if y_preds[i] == y_trues[i] else 0 for i in range(len(y_preds))])
                torch.cuda.empty_cache()
                pbar.set_description(f"Evaluate Reconstruction: Accuracy: {100 * acc:.2f}%")
                pbar.update(1)
    else:
        raise ValueError(f"Unknown attack type: {type(attack)}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
