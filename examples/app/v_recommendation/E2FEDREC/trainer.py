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
from DataSet import DataSet
from secretflow import PYUObject, proxy
from E2FEDREC import Model

import logging


@proxy(PYUObject)
class Trainer:
    """
    Trainer class that handles data loading, model training,
    representation exchange, and evaluation for a single domain.
    """

    def __init__(self, args, dataName, dataName_B, device_id):
        """
        Initialize Trainer.

        Args:
            args: Training arguments
            dataName: Name of the current domain dataset
            dataName_B: Name of the other domain dataset (for cross-domain similarity)
            device_id: GPU device id (integer)
        """
        self.args = args

        # Select computation device (prefer GPU if available)
        if torch.cuda.is_available():
            self.device_id = device_id
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f"cuda:{self.device_id}")
        else:
            self.device_id = -1
            self.device = torch.device("cpu")

        # Load dataset for domain A
        self.dataSet = DataSet(dataName, None)

        # Initialize model (only for domain A)
        self.model = Model(
            args,
            dataName=dataName,
            dataName_B=dataName_B,
            shape=self.dataSet.shape,
            maxRate=self.dataSet.maxRate,
            device_id=device_id,
        )

        # Prepare test set with negative samples
        self.testNeg = self.dataSet.getTestNeg(self.dataSet.test, 99)

        # Training hyperparameters
        self.batch_size = args.batchSize
        self.topK = args.topK
        self.max_epochs = args.maxEpochs
        self.negNum = args.negNum
        self.dataName = dataName
        self.KSize = args.KSize

    def get_reps_shared(self):
        """
        Return user representations to be shared with other clients.

        Returns:
            Cloned tensor of user representations, or None if unavailable.
        """
        reps = self.model.to_B_similarity

        # Check for None case
        if reps is None:
            logging.warning(f"Domain {self.dataName} | to_B_similarity is None.")
            return None

        # Check for empty tensor
        if reps.numel() == 0:
            logging.warning(
                f"Domain {self.dataName} | to_B_similarity is an empty tensor."
            )
            return None

        # Return a copy to avoid accidental modifications
        return reps.clone()

    def set_global_reps(self, global_rep):
        """
        Set global cross-domain representations received from the server.

        Args:
            global_rep: Dictionary mapping dataName to global representations

        Raises:
            AttributeError, TypeError, KeyError
        """
        if not hasattr(self, "model"):
            raise AttributeError("Model not initialized")
        if not isinstance(global_rep, dict):
            raise TypeError("global_rep should be a dictionary")
        if not hasattr(self, "dataName"):
            raise AttributeError("dataName attribute not set")

        try:
            self.model.from_B_similarity = global_rep[self.dataName]
        except KeyError:
            raise KeyError(f"dataName '{self.dataName}' not found in global_rep")

    def run_one_epoch(self, epoch):
        """
        Perform one epoch of training and evaluation.

        Args:
            epoch: Current epoch index (not used inside, reserved for compatibility)

        Returns:
            Tuple (loss, HR@topK, NDCG@topK) for this epoch
        """
        # Prepare training instances (user, item, rating)
        train_u, train_i, train_r = self.dataSet.getInstances(
            self.dataSet.train, self.negNum
        )
        train_len = len(train_u)

        # Shuffle training data
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        # Convert to PyTorch tensors
        train_u = torch.tensor(train_u).to(self.device)
        train_i = torch.tensor(train_i).to(self.device)
        train_r = torch.tensor(train_r).to(self.device)

        # Train model on the batch
        loss = self.model.train_model(
            train_u, train_i, train_r, batch_size=self.batch_size
        )

        # Evaluate model on test data (HR@topK and NDCG@topK)
        hr, ndcg = self.model.evaluate(self.topK, self.testNeg[0], self.testNeg[1])

        # No printing inside this function, return results
        return loss, hr, ndcg
