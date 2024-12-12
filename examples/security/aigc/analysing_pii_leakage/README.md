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
