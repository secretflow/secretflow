import torch
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, BAEGarg2019, TextBuggerLi2018
from SafetyCheckerWrapper import SDPipeline, SDWrapper
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_nsfw_prompts(filepath):
    with open(filepath, 'r') as f:
        data = f.read().split('\n')
    datasets = [(prompt, 1) for prompt in data]
    return datasets, data

datasets, _ = load_nsfw_prompts("../data/nsfw_200.txt")
attack_args = AttackArgs(num_examples=len(datasets), log_to_csv="results/result.csv", csv_coloring_style="plain")
pipeline = SDPipeline(device, mode="ti_sd")

sdwrapper = SDWrapper(pipeline)
attack = BAEGarg2019.build(sdwrapper)
dataset = Dataset(datasets)
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()


