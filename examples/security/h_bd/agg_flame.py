from copy import deepcopy
from typing import List
import numpy as np
import torch
import sklearn.metrics.pairwise as smp
import hdbscan

from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.device import PYU, DeviceObject, PYUObject


class FlameAggregator(PlainAggregator):
    """
    FLAME Aggregator (Byzantine-robust aggregation):
      - data[i]: Local model parameters from each client (list of layer parameters)
      - Clustering phase: Use raw model parameters to build feature vectors,
        filter outliers using HDBSCAN
      - Previous global model estimation: Take element-wise median across clients
        for each layer
      - Gradient clipping: Calculate L2 norms of local updates (client params - estimated global),
        use median norm as clipping threshold
      - Final aggregation: Average clipped parameters with client weights
    """

    def __init__(self, device: PYU):
        """Initialize with specified computation device"""
        assert isinstance(device, PYU), f"Accepts PYU only but got {type(device)}."
        self.device = device

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
        # Move all data to aggregator's device
        data = [d.to(self.device) for d in data]
        if isinstance(weights, (list, tuple)):
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]

        def _average(*data, axis, weights):
            num_clients = len(data)
            num_layers = len(data[0])  # All clients have same layer structure

            # ---------------------------
            # Step 1: Estimate Previous Global Model
            # Element-wise median across clients for each layer
            # ---------------------------
            estimated_global = []
            for layer_idx in range(num_layers):
                layer_params = []
                for i in range(num_clients):
                    arr = data[i][layer_idx]
                    # Convert to numpy array if needed
                    if isinstance(arr, torch.Tensor):
                        arr = arr.cpu().numpy()
                    else:
                        arr = np.array(arr)
                    layer_params.append(arr)
                median_layer = np.median(np.stack(layer_params, axis=0), axis=0)
                estimated_global.append(median_layer)

            # ---------------------------
            # Step 2: Compute Local Updates
            # Update = Client params - Estimated global
            # ---------------------------
            updates = []
            for i in range(num_clients):
                update_i = []
                for layer_idx in range(num_layers):
                    arr = data[i][layer_idx]
                    if isinstance(arr, torch.Tensor):
                        arr = arr.cpu().numpy()
                    else:
                        arr = np.array(arr)
                    update_i.append(arr - estimated_global[layer_idx])
                updates.append(update_i)

            # ---------------------------
            # Step 3: Client Clustering
            # Build feature vectors from raw parameters
            # ---------------------------
            features = []
            for i in range(num_clients):
                flat_model = np.concatenate(
                    [
                        (
                            arr.cpu().numpy()
                            if isinstance(arr, torch.Tensor)
                            else np.array(arr)
                        ).flatten()
                        for arr in data[i]
                    ]
                )
                features.append(flat_model)
            features_array = np.stack(features, axis=0)

            # Compute cosine distance matrix
            cd = smp.cosine_distances(features_array)
            cd = cd.astype(np.float64)  # Ensure HDBSCAN compatibility

            # HDBSCAN clustering configuration
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(num_clients / 2 + 1),  # Require majority cluster
                min_samples=1,
                allow_single_cluster=True,  # Handle all-in-one-cluster case
                metric='precomputed',  # Use precomputed distance matrix
            ).fit(cd)
            cluster_labels = clusterer.labels_.tolist()
            print("Cluster labels:", cluster_labels)

            # ---------------------------
            # Step 4: Compute Update Norms
            # L2 norm of each client's full update vector
            # ---------------------------
            update_norms = []
            for i in range(num_clients):
                flat_update = np.concatenate([upd.flatten() for upd in updates[i]])
                norm_val = np.linalg.norm(flat_update)
                update_norms.append(norm_val)

            # Median norm as clipping threshold
            threshold = np.median(update_norms)

            # ---------------------------
            # Step 5: Norm Clipping
            # Clip updates exceeding threshold and reconstruct models
            # ---------------------------
            clipped_client_models = []
            clipped_weights = []
            for i in range(num_clients):
                # Filter out anomalies (cluster label -1)
                if cluster_labels[i] == -1:
                    continue

                # Calculate scaling factor
                scale = (
                    min(1.0, threshold / update_norms[i])
                    if update_norms[i] > 0
                    else 1.0
                )

                # Apply clipping layer-wise
                clipped_update = []
                for layer_idx in range(num_layers):
                    clipped_update.append(updates[i][layer_idx] * scale)

                # Reconstruct clipped model
                clipped_model = []
                for layer_idx in range(num_layers):
                    clipped_model.append(
                        estimated_global[layer_idx] + clipped_update[layer_idx]
                    )

                clipped_client_models.append(clipped_model)
                clipped_weights.append(weights[i] if weights is not None else 1.0)

            # ---------------------------
            # Step 6: Fallback Handling
            # Return estimated global if all clients filtered
            # ---------------------------
            if not clipped_client_models:
                return estimated_global

            # ---------------------------
            # Step 7: Weighted Aggregation
            # Average clipped models with weights
            # ---------------------------
            aggregated = []
            for layer_idx in range(num_layers):
                layer_vals = [
                    client_model[layer_idx] for client_model in clipped_client_models
                ]
                avg = np.average(layer_vals, axis=0, weights=clipped_weights)
                aggregated.append(avg)

            # ---------------------------
            # Step 8: (Optional) Add Noise
            # Disabled when lambda=0 (current setting)
            # ---------------------------
            final_agg = []
            lamda = 0  # Noise multiplier (0 = no noise)
            for arr in aggregated:
                noise = np.random.normal(
                    loc=0.0, scale=lamda * threshold, size=arr.shape
                )
                final_agg.append(arr + noise)

            return final_agg

        return self.device(_average)(*data, axis=axis, weights=weights)
