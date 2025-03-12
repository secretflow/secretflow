# analysing_pii_leakage

This repository aims to replicating the results in paper  [Analyzing Leakage of Personally Identifiable Information in Language Models.](https://ieeexplore.ieee.org/abstract/document/10179300/). [^1]

[^1]: This repo is a replica of the [analysing_pii_leakage's official implementation](https://github.com/p1kachuu/analysing_pii_leakage). We updated the implementations, such as supporting Parameter-Efficient Fine-Tuning (PEFT) and more advanced Language Models, fixed bugs, and cleaned up the codebase.

This paper introduces rigorous game-based definitions for three types of PII leakage via black-box extraction, inference, and reconstruction attacks with only API access to an LM

This repo is supported by [ANT group](https://www.antgroup.com/) and [Cybersecurity College Student Innovation Funding Program](https://zzjh.org.cn/#/)

## Setup

### Environment
The code is based on python 3.10. It is recommended to use conda create env:

```bash
conda create -n analysing_pii python=3.10
conda activate analysing_pii
```

### Requirements

Dependency can be installed with the following command:

```bash
pip install -r requirements.txt
pip install -e .
cd examples
```


## Usage

We explain the following functions. The scripts are in the ```./examples``` folder and
run configurations are in the ```./configs``` folder.
* **Fine-Tune**: Fine-tune a pre-trained LM on a dataset (optionally FFT or PEFT(LoRa)).
* **PII Extraction**: Given a fine-tuned LM, return a set of PII.
* **PII Reconstruction**: Given a fine-tuned LM and a masked sentence, reconstruct the most likely PII candidate for the masked tokens.
* **PII Inference**: Given a fine-tuned LM, a masked sentence and a set of PII candidates, choose the most likely candidate.

### Tested pretrained models
- GPT-2 Series: gpt2、gpt2-medium、gpt2-large、gpt2-xl (https://huggingface.co/openai-community)
- mistralai/Mistral-7B-Instruct-v0.3 (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- meta-llama/Meta-Llama-3-8B (https://huggingface.co/meta-llama/Meta-Llama-3-8B)


### Tested datasets
- ECHR (https://huggingface.co/datasets/AUEB-NLP/ecthr_cases)
- Enron (https://huggingface.co/datasets/Yale-LILY/aeslc)

**We provide all the tested combinations in the ```./configs``` folder**

## Fine-Tuning

We demonstrate how to fine-tune a ```GPT-2 small``` model on the dataset
(i) FFT, (ii) LoRa, .

**FFT with No Defense**
```shell
accelerate launch python fine_tune.py --config_path ../configs/fine-tune/echr-gpt2-small-undefended.yml 
```

**LoRa with No Defense**
```shell
accelerate launch python fine_tune.py --config_path ../configs/fine-tune/echr-gpt2-small-lora-undefended.yml
```

**With Scrubbing**

_Note_: All PII will be scrubbed from the dataset. Scrubbing is a one-time operation that requires tagging all PII in the dataset first
which can take many hours depending on your setup. We do not provide tagged datasets.
```shell
accelerate launch python fine_tune.py --config_path ../configs/fine-tune/echr-gpt2-small-scrubbed.yml
```


## Attacks

Assuming your fine-tuned model is located at ```../echr_undefended``` run the following attacks.
Otherwise, you can edit the ```model_ckpt``` attribute in the ```../configs/<ATTACK>/echr-gpt2-small-undefended.yml``` file to point to the location of the model.

**PII Extraction**

This will extract PII from the model's generated text.
```shell
python extract_pii.py --config_path ../configs/pii-extraction/echr-gpt2-small-undefended.yml
```

**PII Reconstruction**

This will reconstruct PII from the model given a target sequence.
```shell
python reconstruct_pii.py --config_path ../configs/pii-reconstruction/echr-gpt2-small-undefended.yml
```

**PII Inference**

This will infer PII from the model given a target sequence and a set of PII candidates.
```shell
python reconstruct_pii.py --config_path ../configs/pii-inference/echr-gpt2-small-undefended.yml
```


## Evaluation

Use the ```evaluate.py``` script to evaluate our privacy attacks against the LM.
```shell
python evaluate.py --config_path ../configs/evaluate/echr-gpt2-small-pii-extraction.yml
```
This will compute the precision/recall for PII extraction and accuracy for PII reconstruction/inference attacks.


## Datasets

The provided ECHR dataset wrapper already tags all PII in the dataset.
The PII tagging is done using the Flair NER modules and can take several hours depending on your setup, but is a one-time operation
that will be cached in subsequent runs.


## Fine-Tuned Models

Unfortunately, we do not provide fine-tuned model checkpoints.
This repository does support loading models remotely, which can be done by providing a URL instead of a local path
in the configuration files for the ```model_ckpt``` attribute.


## Optimization Tricks of **PII Reconstruction** Attack
### Sampling Strategy
We have provided and tested the effects of multiple sampling strategies for language models on PII reconstruction. If you want to try it, change the parameters under `sampling_args` in the `configs/evaluate/*-pii-reconstruction.yml`. In addition, you can modify the temperature and repetition penalty to change the probability distribution of the model output. Below we provide the empirical parameters for subsequent experiments of other tricks.
```yml
...

sampling_args:
  top_k: 0.0
  top_p: 0.7
  typical_p: 1.0
  temperature: 0.8
  repetition_penalty: 1.0

...
```

### Member Inference
We explored various methods for improving member inference accuracy in PII reconstruction attacks, focusing on detecting whether generated PII was part of the model's training data.
#### Zlib Compression
This method combines text entropy with perplexity to create a membership inference metric. Zlib compression is used to quantify the information content of a text sequence, providing a measure of entropy. We calculate the ratio or product of model perplexity and Zlib compression score to determine membership likelihood, which has been proven effective[^2]. The intuition is that memorized sequences may have unique entropy and perplexity signatures compared to non-memorized ones. You can try it by setting the `attack_args/attack_name` parameter to `zlib_perplexity_reconstruction` in the config file.

[^2]: Carlini, Nicholas, et al. "Extracting training data from large language models." 30th USENIX security symposium (USENIX Security 21). 2021.


#### Model Perplexity Difference
This approach leverages the difference in perplexity scores between the fine-tuned model and the original pre-trained model. The hypothesis is that samples present in the training data would have lower perplexity in the fine-tuned model but potentially higher perplexity in the original model. We use this perplexity gap as a membership indicator, with larger differences suggesting higher likelihood of training set membership. Setting the `attack_args/attack_name` parameter to `diff_perplexity_reconstruction` in the config file.

#### Substring Perplexity
The substring perplexity method examines not just the perplexity of the full sample but also its substrings. We observed that for genuine training samples, both the full string and its substrings tend to have consistently low perplexity. In contrast, non-training samples might have low perplexity for the full string but higher perplexity for substrings. This pattern can serve as a more reliable membership inference signal than using full-string perplexity alone. You can try this method by setting the `attack_args/attack_name` parameter to `substring_perplexity_reconstruction` in the config file.

### In Context Learning for PII Reconstruction
We enhanced the original PII reconstruction approach by leveraging in-context learning capabilities of language models. Instead of simply using a prefix to prompt the model to complete PII, we experimented with structured prompts in the format of `prefix+<MASK>+suffix`. This provides the model with more contextual information around the missing PII. You can try this by setting the `attack_args/attack_name` parameter to `icl_perplexity_reconstruction` in the config file. Further, we explore the impact of including different numbers of examples in the prompt on performance. You can modify it by changing the `icl_num` (the default is 3).