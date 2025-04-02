import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_attack(k, s, model, dataset, output_path, log_path):
    os.makedirs(output_path, exist_ok=True)

    output_file = f"{output_path}/{dataset}-{model}-k{k}-s{s}.json"
    log_file = f"{log_path}/{dataset}-{model}-k{k}-s{s}.log"
    
    # Command to run
    command = f"""
    python src/attack_poison.py \
       --dataset {dataset}  \
       --split train \
       --model_code {model} \
       --num_cand 100 --per_gpu_eval_batch_size 64 --num_iter 5000 --num_grad_iter 1 \
       --output_file {output_file} \
       --do_kmeans --k {k} --kmeans_split {s}
    """
    
    # Open log file for writing
    with open(log_file, "w") as log:
        subprocess.run(command, shell=True, check=True, stdout=log, stderr=log)

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

if __name__ == "__main__":
    main()
