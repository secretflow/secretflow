# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import argparse
import secretflow as sf

from E2FEDREC import Model
from trainer import Trainer
from server import Server
from secretflow import PYUObject, proxy


def get_args(K_size):
    """
    Parse and return command-line arguments for training configuration.

    Args:
        K_size (int): Initial embedding dimension size

    Returns:
        argparse.Namespace: Parsed training arguments
    """
    parser = argparse.ArgumentParser(
        description="Cross-Domain Recommendation System Options"
    )

    # Training hyperparameters
    parser.add_argument(
        "-negNum",
        default=7,
        type=int,
        help="Number of negative samples per positive sample",
    )
    parser.add_argument("-lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "-maxEpochs", default=20, type=int, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "-batchSize", default=512, type=int, help="Mini-batch size for training"
    )
    parser.add_argument(
        "-earlyStop", default=5, type=int, help="Early stopping patience"
    )
    parser.add_argument(
        "-checkPoint", default="./checkPoint/", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "-topK", default=10, type=int, help="Top-K value for evaluation metrics"
    )

    # Model architecture configuration
    parser.add_argument(
        "-userLayer",
        default=[K_size, 2 * K_size, K_size],
        help="Sizes of user encoder layers",
    )
    parser.add_argument(
        "-itemLayer",
        default=[K_size, 2 * K_size, K_size],
        help="Sizes of item encoder layers",
    )
    parser.add_argument("-KSize", default=K_size, help="Embedding dimension size")

    # Regularization settings
    parser.add_argument(
        "-reg", default=1e-3, type=float, help="L2 regularization coefficient"
    )
    parser.add_argument(
        "-lambdad", default=0.001, type=float, help="Weight for domain alignment loss"
    )

    # Contrastive learning settings
    parser.add_argument(
        "-ssl_temp", default=1, type=float, help="Temperature for contrastive loss"
    )
    parser.add_argument(
        "-ssl_reg_intra",
        default=0.3,
        type=float,
        help="Weight for intra-domain contrastive loss",
    )
    parser.add_argument(
        "-ssl_reg_inter",
        default=0.2,
        type=float,
        help="Weight for inter-domain contrastive loss",
    )

    # Empty list to prevent CLI parsing conflict when called from code
    return parser.parse_args([])


def train_epochs(records, server, max_epochs):
    """
    Train models across domains for a given number of epochs.

    Args:
        records (dict): Dictionary storing trainer instances and metrics
        server (Server): Server instance for aggregation
        max_epochs (int): Maximum number of training epochs
    """
    print("[INFO] Starting epoch training...")

    dataset_names = list(records.keys())

    for epoch in range(max_epochs):
        print(f"\n[INFO] === Epoch {epoch} ===")

        # Step 1: Collect representations from each trainer
        trainer_weights = []
        for dataset_name in dataset_names:
            trainer = records[dataset_name]["trainer"]
            weights = trainer.get_reps_shared()
            if weights is None:
                raise ValueError(
                    f"[ERROR] {dataset_name}: get_reps_shared() returned None."
                )
            trainer_weights.append(weights.to(server.device))

        # Step 2: Aggregate representations at the server
        global_weights = server.aggregate_reps(trainer_weights)

        # Step 3: Distribute global representations back to trainers
        setting = []
        for dataset_name in dataset_names:
            trainer = records[dataset_name]["trainer"]
            ret = trainer.set_global_reps(global_weights.to(trainer.device))
            setting.append(ret)

        sf.wait(setting)  # Ensure all trainers have synchronized

        # Step 4: Train and evaluate
        for dataset_name in dataset_names:
            trainer = records[dataset_name]["trainer"]
            print(f"[INFO] Training for {dataset_name}...")
            loss, hr, ndcg = sf.reveal(trainer.run_one_epoch(epoch))

            records[dataset_name]["loss_list"].append(loss)
            records[dataset_name]["hr_list"].append(hr)
            records[dataset_name]["NDCG_list"].append(ndcg)

            if hr > records[dataset_name]["best_hr"]:
                records[dataset_name]["best_hr"] = hr
                records[dataset_name]["best_NDCG"] = ndcg

            print(
                f"[RESULT] {dataset_name} | Epoch {epoch} | Loss: {loss:.4f}, HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}"
            )

    print("[INFO] Epoch training completed.")


def main(dataName_A, dataName_B, K_size):
    """
    Main entry point for running cross-domain federated training.

    Args:
        dataName_A (str): Name of source domain dataset
        dataName_B (str): Name of target domain dataset
        K_size (int): Embedding dimension size
    """
    print(
        f"[INFO] Starting training with:\n  - Domain A: {dataName_A}\n  - Domain B: {dataName_B}\n  - K size: {K_size}"
    )

    try:
        # === Setup section ===
        args = get_args(K_size)
        np.random.seed(42)

        sf.shutdown()  # Clean up any previous session (optional)
        sf.init([dataName_A, dataName_B, "server"], address="local", num_gpus=1)
        print("[INFO] SecretFlow initialized successfully.")

        dataName_A_pyu = sf.PYU(dataName_A)
        dataName_B_pyu = sf.PYU(dataName_B)
        server_pyu = sf.PYU("server")

        # Create trainers and server
        trainer_A = Trainer(args, dataName_A, dataName_B, 0, device=dataName_A_pyu)
        trainer_B = Trainer(args, dataName_B, dataName_A, 0, device=dataName_B_pyu)
        server = Server(args, [dataName_A, dataName_B], device=server_pyu)

        # Initialize metrics recorder
        records = {
            dataName_A: {
                "trainer": trainer_A,
                "loss_list": [],
                "hr_list": [],
                "NDCG_list": [],
                "best_hr": -1,
                "best_NDCG": -1,
            },
            dataName_B: {
                "trainer": trainer_B,
                "loss_list": [],
                "hr_list": [],
                "NDCG_list": [],
                "best_hr": -1,
                "best_NDCG": -1,
            },
        }

        # === Training section ===
        max_epochs = args.maxEpochs
        train_epochs(records, server, max_epochs)

        # Save results
        for dataset_name in [dataName_A, dataName_B]:
            trainer = records[dataset_name]["trainer"]
            matname = f"E2FEDREC_{dataset_name}_KSize_{K_size}_Result.pt"
            torch.save(
                {
                    "loss_list": records[dataset_name]["loss_list"],
                    "hr_list": records[dataset_name]["hr_list"],
                    "NDCG_list": records[dataset_name]["NDCG_list"],
                    "bestPerformance": [
                        records[dataset_name]["best_hr"],
                        records[dataset_name]["best_NDCG"],
                    ],
                },
                matname,
            )
            print(f"[INFO] Results for {dataset_name} saved to {matname}")

        print("[INFO] Training completed successfully.")

    except Exception as e:
        print("[ERROR] Exception occurred during training:")
        print(e)

    print("[INFO] Program execution finished.")


if __name__ == "__main__":
    tasks = [["book", "movie"]]
    KList = [8]

    for K_size in KList:
        for domain_A, domain_B in tasks:
            main(domain_A, domain_B, K_size)
