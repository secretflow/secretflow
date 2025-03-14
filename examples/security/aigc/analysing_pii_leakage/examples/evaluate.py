# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import torch
import numpy as np
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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

        # add ten-shot prompt (icl_perplexity_reconstruction needs)
        prompts = []
        targets = []
        for i in range(100):
            if len(prompts) >= 10:
                break
            seq = dataset[-i]
            pii = tagger.analyze(seq['text']).get_by_entity_class(attack_args.pii_class).unique()
            pii = ListPII(data=[p for p in pii if len(p.text) > 3 and p.start > 0])  # min chars for PII
            if len(pii) == 0:
                continue
            target_pii = random.sample(pii.mentions(), 1)[0]
            if target_pii not in seq['text']:
                continue

            target_sequence = seq['text'].replace(target_pii, '<T-MASK>', 1)
            prefix, suffix = target_sequence.split("<T-MASK>")

            prefix = ' '.join(prefix.split(' ')[-20:])
            suffix = ' '.join(suffix.split(' ')[:20])

            prompt_seq = prefix + "<T-MASK>" + suffix
            if target_pii in prompt_seq:
                continue

            prompts.append(prompt_seq)
            targets.append(target_pii)


        with tqdm(total=eval_args.num_sequences, desc="Evaluate Reconstruction") as pbar:
            y_preds, y_trues = [], []
            y_results = []

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
                # for pii_mention in pii.mentions():
                #     target_sequence = target_sequence.replace(pii_mention, '<MASK>')

                # 4. Randomly sample candidates
                assert eval_args.num_candidates <= len(real_pii.unique().mentions()), f"Not enough real candidates " \
                                                                                        f"({len(real_pii.unique().mentions())}) to accomodate candidate size ({eval_args.num_candidates})."
                candidate_pii = random.sample(real_pii.unique().mentions(), eval_args.num_candidates - 1) + [
                    target_pii]
                random.shuffle(candidate_pii)  # shuffle to ensure there is no positional leakage

                # 5. Run the reconstruction attack
                if attack_args.attack_name == 'icl_perplexity_reconstruction':
                    result = attack.attack(baseline_lm, lm, [prompts, targets], target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                elif attack_args.attack_name == 'diff_perplexity_inference' or attack_args.attack_name == 'diff_perplexity_reconstruction':
                    results = attack.attack(baseline_lm, lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                    if not results:
                        continue
                    ppl, ppl_baselines, result, candidates = results[0], results[1], results[2], results[3]
                    # 添加可视化
                    if target_pii in candidates:
                        plot_path = plot_perplexity_comparison(ppl, ppl_baselines, candidates, target_pii)
                        print(f"Perplexity comparison plot saved to: {plot_path}\n")
                else:
                    result = attack.attack(baseline_lm, lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)

                if not result:
                    continue
                if attack_args.attack_name == 'diff_perplexity_inference' or attack_args.attack_name == 'diff_perplexity_reconstruction':
                    # 计算困惑度差值
                    if isinstance(ppl, torch.Tensor):
                        ppl = ppl.float().cpu().numpy()
                        ppl_baselines = ppl_baselines.float().cpu().numpy()
                    ppl_diff = ppl - ppl_baselines
                    
                    # 获取困惑度最低的前k个候选项
                    num_top = attack_args.diff_topk
                    top5_indices = np.argsort(ppl)[:num_top]
                    
                    # 从这5个候选项中选择困惑度差值最小的
                    top5_diffs = ppl_diff[top5_indices]
                    best_idx = top5_indices[np.argmin(top5_diffs)]
                    predicted_target_pii = candidates[best_idx]
                else:
                    predicted_target_pii = result[min(result.keys())]

                # 6. Evaluate baseline leakage
                # baseline_result = attack.attack(baseline_lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                # baseline_target_pii = baseline_result[min(baseline_result.keys())]

                # if baseline_target_pii == predicted_target_pii:
                #     # Baseline leakage because public model has the same prediction. Skip
                #     continue

                y_preds += [predicted_target_pii]
                y_results.append(result.values())
                y_trues += [target_pii]

                acc = np.mean([1 if y_preds[i] == y_trues[i] else 0 for i in range(len(y_preds))])
                acc_recall = np.mean([1 if y_trues[i] in y_results[i] else 0 for i in range(len(y_results))])
                torch.cuda.empty_cache()
                pbar.set_description(f"Evaluate Reconstruction: Accuracy: {100 * acc:.2f}%, Acc_recall: {100 * acc_recall:.2f}%")
                pbar.update(1)
    else:
        raise ValueError(f"Unknown attack type: {type(attack)}")

def plot_perplexity_comparison(ppl, ppl_baselines, candidate_pii, target_pii, save_dir='plots'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将tensor转换为numpy数组并确保数据类型为float32
    if isinstance(ppl, torch.Tensor):
        ppl = ppl.float().cpu().numpy()
        ppl_baselines = ppl_baselines.float().cpu().numpy()
    
    # 计算差值并获取排序索引
    ppl_diff = ppl - ppl_baselines
    sort_indices = np.argsort(ppl_diff)
    
    # 按差值排序所有数据
    ppl_sorted = ppl[sort_indices]
    ppl_baselines_sorted = ppl_baselines[sort_indices]
    candidate_pii_sorted = [candidate_pii[i] for i in sort_indices]
    ppl_diff_sorted = ppl_diff[sort_indices]
    
    # 设置图表样式
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    x = range(len(candidate_pii_sorted))
    width = 0.35
    
    # 绘制柱状图
    bars1 = plt.bar(x, ppl_sorted, width, label='Fine-tuned Model', alpha=0.8)
    bars2 = plt.bar([i + width for i in x], ppl_baselines_sorted, width, label='Baseline Model', alpha=0.8)
    
    # 设置标签和标题
    plt.xlabel('Candidate PII (sorted by perplexity difference)')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison between Models')
    plt.xticks([i + width/2 for i in x], candidate_pii_sorted, rotation=45, ha='right')
    
    # 添加图例
    plt.legend()
    
    # 高亮目标PII
    target_idx = candidate_pii_sorted.index(target_pii)
    bars1[target_idx].set_color('red')
    bars2[target_idx].set_color('red')
    
    
    # 添加目标PII标注
    plt.annotate('Target PII', 
                xy=(target_idx + width/2, max(ppl_sorted[target_idx], ppl_baselines_sorted[target_idx])),
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f'perplexity_comparison_{len(os.listdir(save_dir))}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
