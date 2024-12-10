# Corpus-poisoning

This repository aims to replicating the results in paper  [Poisoning Retrieval Corpora by Injecting Adversarial Passages](https://arxiv.org/abs/2310.19156).

This paper propose the *corpus poisoning* attack for dense retrieval models, where a malicious user generates and injects a small fraction of adversarial passages to a retrieval corpus, with the aim of fooling retrieval systems into returning them among the top retrieved results.


This repo is supported by [ANT group](https://www.antgroup.com/) and [Cybersecurity College Student Innovation Funding Program](https://zzjh.org.cn/#/)

## Setup

### Create a conda environment
The code is based on python 3.7. It is recommended to use conda

```bash
conda create -n corpus_poison python=3.7
conda activate corpus_poison
```


### Requirements
Please install all the dependency packages using the following command:
```bash
pip install -r requirements.txt
```
### contriever

The code needs contriever support, the reproducible code is already included.

place it into the `src` folder:
```bash
cd src
git clone https://github.com/facebookresearch/contriever.git
```

### Datasets

There is a tutorial for installing the dataset under the [Datasets](./datasets/README.md)

## Attack: Corpus poisoning

The sample generation process of the adversarial attack, the core attack steps.

- Run through the one-click run file [`run.py`](./run.py)
- `run.py` supports multi-threaded operation
- `run.py` supports running records and logs
```python
def main():
    model = "ance"
    dataset = "nq-train"
    k = 10
    output_path = "results/advp"
    log_path = "logs"  # Specify the directory for log files

    # Ensure the log directory exists
    os.makedirs(log_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for s in range(k):
            futures.append(executor.submit(run_attack, k, s, model, dataset, output_path, log_path))

        # Wait for all threads to complete
        for future in futures:
            future.result()
```

- If you want to run it alone, execute the following code:
```bash
python src/attack_poison.py \
   --dataset ${DATASET} --split train \
   --model_code ${MODEL} \
   --num_cand 100 --per_gpu_eval_batch_size 64 --num_iter 5000 --num_grad_iter 1 \
   --output_file ${OUTPUT_PATH}/${DATASET}-${MODEL}-k${k}-s${s}.json \
   --do_kmeans --k $k --kmeans_split $s
```

## Evaluate the attack

After running the attack, we generate a set of adversarial examples (by default, they are saved in `results/advp`). Here, we evaluate the attack performance.

We first perform the original retrieval evaluation on BEIR and save the retrieval results (i.e., for a given query, we save a list of top passages and similarity values to the query). Then, we evaluate the attack by checking if the similarity of any adversarial passage to the query is greater than that of the top-20 passages on the original corpus.

### BEIR evaluation

Get the initial ranking and save the ranking results of the retriever.

```bash
MODEL=contriever
DATASET=fiqa

mkdir -p results/beir_results
python src/evaluate_beir.py --model_code contriever --dataset fiqa --result_output results/beir_results/${DATASET}-${MODEL}.json
```

Evaluation is available for models (`contriever`, `contriever-msmarco`, `dpr-single`, `dpr-multi`, `ance`) and datasets (`nq`, `msmarco`, `hotpotqa`, `fiqa`, `trec-covid`, `nfcorpus`, `arguana`, `quora`, `scidocs`, `fever`, `scifact`)

### Adversarial attack evaluation

Then, we evaluate the attack based on the beir retrieval results (saved in `results/beir_results`) and the generated adversarial passages (saved in `results/advp`).

Execute the following code to obtain the evaluation results.

```bash
EVAL_MODEL=contriever
EVAL_DATASET=fiqa
ATTK_MODEL=contriever
ATTK_DATASET=nq-train

python src/evaluate_adv.py --save_results \
   --attack_model_code ${ATTK_MODEL} --attack_dataset ${ATTK_DATASET} \
   --advp_path results/advp --num_advp 10 \
   --eval_model_code ${EVAL_MODEL} --eval_dataset ${EVAL_DATASET} \
   --orig_beir_results results/beir_results/${EVAL_DATASET}-${EVAL_MODEL}.json 
```

## result

All running results are saved in `result`.

More explanation of the results is in [result](./results/README.md)

