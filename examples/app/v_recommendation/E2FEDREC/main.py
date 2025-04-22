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

import numpy as np
import argparse
from datetime import datetime
from E2FEDREC import Trainer
from E2FEDREC import Model
import secretflow as sf
from secretflow import PYUObject, proxy


def get_args(dataName_A, dataName_B, K_size):
    """
    Parses command-line arguments and returns configuration parameters.

    Args:
        dataName_A (str): Source domain name
        dataName_B (str): Target domain name
        K_size (int): Initial embedding dimension size

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Cross-Domain Recommendation System Options")

    # Dataset configuration
    parser.add_argument('-dataName_A', default=dataName_A,
                        help='Name of source domain dataset')
    parser.add_argument('-dataName_B', default=dataName_B,
                        help='Name of target domain dataset')

    # Training configuration
    parser.add_argument('-negNum', default=7, type=int,
                        help='Number of negative samples per positive sample')
    parser.add_argument('-lr', default=0.001, type=float,
                        help='Learning rate for optimization')
    parser.add_argument('-maxEpochs', default=20, type=int,
                        help='Maximum training epochs')
    parser.add_argument('-batchSize', default=512,
                        type=int, help='Training batch size')
    parser.add_argument('-earlyStop', default=5, type=int,
                        help='Early stopping patience')
    parser.add_argument('-checkPoint', default='./checkPoint/',
                        help='Model checkpoint directory')
    parser.add_argument('-topK', default=10, type=int,
                        help='Top-K for evaluation metrics')

    # Model architecture
    parser.add_argument(
        '-userLayer', default=[K_size, 2 * K_size, K_size], help='User encoder layer sizes')
    parser.add_argument(
        '-itemLayer', default=[K_size, 2 * K_size, K_size], help='Item encoder layer sizes')
    parser.add_argument('-KSize', default=K_size,
                        help='Embedding dimension size')

    # Regularization
    parser.add_argument('-reg', default=1e-3, type=float,
                        help='L2 regularization coefficient')
    parser.add_argument('-lambdad', default=0.001, type=float,
                        help='Domain alignment loss weight')

    # SSL loss
    parser.add_argument('-ssl_temp', default=1, type=float,
                        help='Contrastive loss temperature')
    parser.add_argument('-ssl_reg_intra', default=0.3,
                        type=float, help='Intra-domain loss weight')
    parser.add_argument('-ssl_reg_inter', default=0.2,
                        type=float, help='Inter-domain loss weight')

    # empty list to avoid CLI conflict when called programmatically
    return parser.parse_args([])


def main(dataName_A, dataName_B, K_size):
    """
    Main function to configure and run the cross-domain recommendation model.
    """
    print(
        f"[INFO] Starting training with:\n  - Domain A: {dataName_A}\n  - Domain B: {dataName_B}\n  - K size: {K_size}")

    args = get_args(dataName_A, dataName_B, K_size)
    np.random.seed(42)

    print("[INFO] Initializing SecretFlow environment...")
    sf.shutdown()  # 清理旧状态（可选）
    sf.init(["alice"], address='local', num_gpus=1)
    print("[INFO] SecretFlow initialized successfully.")

    alice_pyu = sf.PYU("alice")
    print("[INFO] Creating Trainer instance on device: alice_pyu")

    trainer = Trainer(args, 0, device=alice_pyu)

    print("[INFO] Starting trainer.run() execution...")
    try:
        result = sf.reveal(trainer.run())
        print("[INFO] trainer.run() completed successfully.")
    except Exception as e:
        print("[ERROR] Exception occurred during trainer.run():")
        print(e)

    print("[INFO] Program execution finished.")


if __name__ == '__main__':
    tasks = [
        ['book', 'movie']
    ]
    KList = [8]

    for K_Size in KList:
        for domain_A, domain_B in tasks:
            main(domain_A, domain_B, K_Size)
