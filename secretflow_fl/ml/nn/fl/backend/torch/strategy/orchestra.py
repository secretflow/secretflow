"""
Orchestra Strategy for Unsupervised Federated Learning

Implementation of Orchestra strategy following the original paper:
"Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering"

Key features:
- Unsupervised federated learning
- Global consistency clustering
- Contrastive learning with EMA target model
- Sinkhorn-Knopp equal-size clustering
- Rotation prediction for degeneracy regularization
- Local and global clustering mechanisms
"""

import copy
from typing import Dict, Tuple, List, Optional
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


def sknopp(cZ, lamd=25, max_iters=100, num_iters=None):
    """
    Sinkhorn-Knopp algorithm for equal-size clustering
    
    Args:
        cZ: Similarity matrix [N_samples, N_centroids]
        lamd: Temperature parameter for softmax
        max_iters: Maximum number of iterations
        num_iters: Specific number of iterations (if provided, overrides max_iters)
    
    Returns:
        Soft cluster assignments [N_samples, N_centroids]
    """
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape
        probs = F.softmax(cZ * lamd, dim=1).T  # [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        iters = num_iters if num_iters is not None else max_iters
        
        for it in range(iters):
            r = inv_N_centroids / (probs @ c)
            c_new = inv_N_samples / (r.T @ probs).T
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if num_iters is None and err < 1e-2:
                break

        probs *= c.squeeze()
        probs = probs.T  # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples


class ProjectionMLP(nn.Module):
    """
    Projection MLP for Orchestra following the original paper architecture
    """
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(ProjectionMLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x)))
        x = self.layer2_bn(self.layer2(x))
        return x


@register_strategy(strategy_name='orchestra', backend='torch')
class OrchestraStrategy(BaseTorchModel):
    """
    Orchestra Strategy for Unsupervised Federated Learning
    
    Implementation following the original paper architecture with:
    - Backbone network for feature extraction
    - Projection network for contrastive learning
    - Target network with EMA updates
    - Global and local clustering centers
    - Sinkhorn-Knopp equal-size clustering
    - Rotation prediction for degeneracy regularization
    
    Args:
        builder_base: Model builder for backbone network
        num_classes: Number of output classes
        temperature: Temperature parameter for contrastive learning (default: 0.1)
        cluster_weight: Weight for clustering loss (default: 1.0)
        contrastive_weight: Weight for contrastive loss (default: 1.0)
        deg_weight: Weight for degeneracy regularization loss (default: 0.1)
        ema_decay: EMA decay rate for target model updates (default: 0.999)
        num_local_clusters: Number of local clusters (N_local, default: 20)
        num_global_clusters: Number of global clusters (N_global, default: 10)
        memory_size: Size of projection memory bank (default: 1024)
        projection_dim: Dimension of projection space (default: 512)
        hidden_dim: Hidden dimension for projection MLP (default: 512)
        epsilon: Epsilon parameter for Sinkhorn-Knopp algorithm (default: 0.05)
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations (default: 3)
    """
    
    def __init__(
        self,
        builder_base,
        num_classes: int = 10,
        temperature: float = 0.1,
        cluster_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        deg_weight: float = 0.1,
        ema_decay: float = 0.999,
        num_local_clusters: int = 20,
        num_global_clusters: int = 10,
        memory_size: int = 1024,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        **kwargs
    ):
        super().__init__(builder_base=builder_base, **kwargs)
        
        # Orchestra parameters following the original paper
        self.temperature = temperature
        self.cluster_weight = cluster_weight
        self.contrastive_weight = contrastive_weight
        self.deg_weight = deg_weight
        self.ema_decay = ema_decay
        self.N_local = num_local_clusters  # Number of local clusters
        self.N_global = num_global_clusters  # Number of global clusters
        self.memory_size = memory_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.round_count = 0
        
        # Initialize Orchestra components
        self._initialize_orchestra_components()
        
    def _initialize_orchestra_components(self):
        """
        Initialize Orchestra components following the original paper architecture
        """
        # Set backbone model and modify it for feature extraction
        self.backbone = self.model
        
        # If the model has a classifier/fc layer, we need to extract features before it
        if hasattr(self.backbone, 'fc'):
            # Store the original fc layer and replace with identity
            self.original_fc = self.backbone.fc
            self.backbone.fc = nn.Identity()
            self.logger.info("Replaced backbone.fc with Identity for feature extraction")
        elif hasattr(self.backbone, 'classifier'):
            # Store the original classifier and replace with identity
            self.original_classifier = self.backbone.classifier
            self.backbone.classifier = nn.Identity()
            self.logger.info("Replaced backbone.classifier with Identity for feature extraction")
        
        # Get backbone output dimension
        backbone_dim = None
        
        # Method 1: Check for explicit output_dim attribute
        if hasattr(self.backbone, 'output_dim'):
            backbone_dim = self.backbone.output_dim
            self.logger.info(f"Found backbone.output_dim: {backbone_dim}")
        
        # Method 2: Check for original fc layer (before replacement)
        elif hasattr(self, 'original_fc') and hasattr(self.original_fc, 'in_features'):
            backbone_dim = self.original_fc.in_features
            self.logger.info(f"Found original_fc.in_features: {backbone_dim}")
        
        # Method 3: Check for original classifier layer (before replacement)
        elif hasattr(self, 'original_classifier') and hasattr(self.original_classifier, 'in_features'):
            backbone_dim = self.original_classifier.in_features
            self.logger.info(f"Found original_classifier.in_features: {backbone_dim}")
        
        # Method 4: Try to infer from a forward pass
        if backbone_dim is None:
            try:
                # Try different input sizes
                test_inputs = [
                    torch.randn(1, 3, 32, 32),  # CIFAR-10 size
                    torch.randn(1, 3, 224, 224),  # ImageNet size
                    torch.randn(1, 1, 28, 28),  # MNIST size
                ]
                
                for dummy_input in test_inputs:
                    try:
                        with torch.no_grad():
                            dummy_output = self.backbone(dummy_input)
                            if len(dummy_output.shape) == 2:  # [batch_size, features]
                                backbone_dim = dummy_output.shape[-1]
                                self.logger.info(f"Inferred backbone output dim from forward pass: {backbone_dim} with input shape {dummy_input.shape}")
                                break
                            elif len(dummy_output.shape) == 4:  # [batch_size, channels, height, width]
                                # Apply global average pooling
                                pooled_output = torch.mean(dummy_output, dim=[2, 3])
                                backbone_dim = pooled_output.shape[-1]
                                self.logger.info(f"Inferred backbone output dim after GAP: {backbone_dim} with input shape {dummy_input.shape}")
                                break
                    except Exception as e:
                        self.logger.debug(f"Failed to test input shape {dummy_input.shape}: {e}")
                        continue
            except Exception as e:
                self.logger.error(f"Failed to infer backbone output dimension: {e}")
        
        # Fallback to a reasonable default
        if backbone_dim is None:
            backbone_dim = 512
            self.logger.warning(f"Could not determine backbone output dimension, using default: {backbone_dim}")
        
        # Initialize projection network
        self.projector = ProjectionMLP(
            in_dim=backbone_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.projection_dim
        )
        
        # Initialize target backbone (EMA copy)
        self.target_backbone = copy.deepcopy(self.backbone)
        for param in self.target_backbone.parameters():
            param.requires_grad = False
            
        # Initialize target projector (EMA copy)
        self.target_projector = copy.deepcopy(self.projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        # Degeneracy regularization layer (rotation prediction)
        self.deg_layer = nn.Linear(self.projection_dim, 4)
        
        # Initialize clustering components following the original paper
        self.mem_projections = nn.Linear(self.memory_size, self.projection_dim, bias=False)
        self.centroids = nn.Linear(self.projection_dim, self.N_global, bias=False)
        self.local_centroids = nn.Linear(self.projection_dim, self.N_local, bias=False)
        
        # Move all components to the correct device
        device = self.exe_device if hasattr(self, 'exe_device') else torch.device('cpu')
        self.backbone = self.backbone.to(device)
        self.target_backbone = self.target_backbone.to(device)
        self.projector = self.projector.to(device)
        self.target_projector = self.target_projector.to(device)
        self.deg_layer = self.deg_layer.to(device)
        self.mem_projections = self.mem_projections.to(device)
        self.centroids = self.centroids.to(device)
        self.local_centroids = self.local_centroids.to(device)
        
        self.logger.info(f"Orchestra components initialized successfully on device: {device}")
    
    def _update_target_model(self):
        """
        Update target model using EMA following the original paper
        """
        with torch.no_grad():
            tau = self.ema_decay
            # Update target backbone
            for target_param, online_param in zip(
                self.target_backbone.parameters(), 
                self.backbone.parameters()
            ):
                target_param.data = tau * target_param.data + (1 - tau) * online_param.data
            
            # Update target projector
            for target_param, online_param in zip(
                self.target_projector.parameters(), 
                self.projector.parameters()
            ):
                target_param.data = tau * target_param.data + (1 - tau) * online_param.data
    
    def _reset_memory(self, dataloader, device='cpu'):
        """
        Reset projection memory following the original paper 
        """
        self.train()
        
        # Save BN parameters 
        reset_dict = OrderedDict()
        for k, v in self.state_dict().items():
            if 'bn' in k:
                if v.shape == ():
                    reset_dict[k] = torch.tensor(np.array([v.cpu().numpy()]))
                else:
                    reset_dict[k] = v.cpu().clone()  
        
        # Generate feature bank with memory optimization
        proj_bank = []
        n_samples = 0
        batch_count = 0
        max_batches = min(10, len(dataloader))  
        
        with torch.no_grad():
            for batch in dataloader:
                if n_samples >= self.memory_size or batch_count >= max_batches:
                    break
                
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                # Limit the batch size to reduce memory usage
                if x.shape[0] > 32:
                    x = x[:32]
                
                x = x.to(device)
                # Use target model for memory initialization
                features = self.target_backbone(x)
                z = F.normalize(self.target_projector(features), dim=1)
                proj_bank.append(z.cpu())  # Move to CPU to reduce GPU memory usage
                n_samples += x.shape[0]
                batch_count += 1
                
                del x, features, z
        
        # Concatenate and truncate if necessary
        if proj_bank:
            proj_bank = torch.cat(proj_bank, dim=0).contiguous()
            if n_samples > self.memory_size:
                proj_bank = proj_bank[:self.memory_size]
            
            # Save projections: [D, memory_size]
            self.mem_projections.weight.data.copy_(proj_bank.T.to(device))
            
            del proj_bank
        
        # Reset BN parameters
        self.load_state_dict(reset_dict, strict=False)
        
        del reset_dict
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def _update_memory(self, features):
        """
        Update projection memory following the original paper 
        """
        N = features.shape[0]
        with torch.no_grad():
            # Optimization: Use in-place operations to avoid creating new tensor copies
            if N < self.memory_size:
                # Shift memory using in-place operations
                self.mem_projections.weight.data[:, :-N].copy_(self.mem_projections.weight.data[:, N:])
                # Add new features using in-place copy
                self.mem_projections.weight.data[:, -N:].copy_(features.T)
            else:
                # If batch size >= memory size, replace entire memory
                self.mem_projections.weight.data.copy_(features[-self.memory_size:].T)
            
            del features
    
    def _local_clustering(self, device='cpu'):
        """
        Perform local clustering following the original paper 
        """
        with torch.no_grad():
            # Get memory projections: [memory_size, D] 
            Z = self.mem_projections.weight.data.T
            
            # Initialize centroids randomly
            indices = np.random.choice(Z.shape[0], self.N_local, replace=False)
            centroids = Z[indices].detach().clone()  # Only clone when necessary
            
            # Local clustering iterations 
            local_iters = 3  # 从5减少到3
            for it in range(local_iters):
                # Compute assignments using Sinkhorn-Knopp
                assigns = sknopp(Z @ centroids.T, max_iters=5)  # Reduce Sinkhorn iterations
                choice_cluster = torch.argmax(assigns, dim=1)
                
                # Update centroids with memory optimization
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    if selected.numel() == 0:
                        selected = torch.randint(len(Z), (1,), device=Z.device)
                    elif selected.dim() == 0:
                        selected = selected.unsqueeze(0)
                    
                    selected_features = torch.index_select(Z, 0, selected)
                    if selected_features.shape[0] == 0:
                        selected_features = Z[torch.randint(len(Z), (1,), device=Z.device)]
                    
                    # Update the centroid using in-place operations
                    centroids[index].copy_(F.normalize(selected_features.mean(dim=0), dim=0))
                
                del assigns, choice_cluster
            
            # Save local centroids
            self.local_centroids.weight.data.copy_(centroids.to(device))
            
            del centroids, Z
    
    def _global_clustering(self, aggregated_features, device='cpu'):
        """
        Perform global clustering on the server following the original paper - 优化内存使用
        
        Args:
            aggregated_features: Aggregated local centroids from all clients
        """
        N = aggregated_features.shape[0]
        
        # Limit the number of aggregated features to reduce memory usage
        if N > 1000:
            indices_sample = torch.randperm(N)[:1000]
            aggregated_features = aggregated_features[indices_sample]
            N = 1000
        
        # Initialize global centroids if not exists
        if not hasattr(self, '_global_centroids_initialized'):
            with torch.no_grad():
                indices = np.random.choice(N, min(self.N_global, N), replace=False)
                init_centroids = aggregated_features[indices]
                self.centroids.weight.data.copy_(init_centroids.T)
                self._global_centroids_initialized = True
        
        # Setup optimizer for global clustering
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
        total_rounds = 100  
        train_loss = 0.
        
        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Compute Sinkhorn-Knopp assignments
                SK_assigns = sknopp(self.centroids(aggregated_features), max_iters=5)  # 减少Sinkhorn迭代
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute predicted assignments
            normalized_features = F.normalize(aggregated_features, dim=1)
            probs = F.softmax(self.centroids(normalized_features) / self.temperature, dim=1)
            
            # Compute loss
            loss = -F.cosine_similarity(SK_assigns, probs, dim=-1).mean()
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Normalize centroids
            with torch.no_grad():
                self.centroids.weight.data = F.normalize(self.centroids.weight.data, dim=1)
                train_loss += loss.item()
            
            # Clean up temporary variables
            del SK_assigns, normalized_features, probs, loss
            
            # Early stopping mechanism
            if round_idx > 20 and round_idx % 10 == 0:
                if train_loss / (round_idx + 1) < 0.01:
                    break
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _generate_rotation_data(self, x):
        """
        Generate rotation data for degeneracy regularization
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (rotated_data, rotation_labels)
        """
        batch_size = x.shape[0]
        
        # Generate random rotation labels (0, 1, 2, 3 for 0°, 90°, 180°, 270°)
        rotation_labels = torch.randint(0, 4, (batch_size,), device=x.device)
        
        # Apply rotations
        rotated_data = torch.zeros_like(x)
        for i, label in enumerate(rotation_labels):
            if label == 0:  # 0°
                rotated_data[i] = x[i]
            elif label == 1:  # 90°
                rotated_data[i] = torch.rot90(x[i], k=1, dims=[-2, -1])
            elif label == 2:  # 180°
                rotated_data[i] = torch.rot90(x[i], k=2, dims=[-2, -1])
            elif label == 3:  # 270°
                rotated_data[i] = torch.rot90(x[i], k=3, dims=[-2, -1])
        
        return rotated_data, rotation_labels
    
    def train_step(self, weights, cur_steps: int, train_steps: int, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Perform Orchestra training steps
        
        Args:
            weights: Global weights from parameter server
            cur_steps: Current training step
            train_steps: Number of local training steps
            **kwargs: Additional arguments including data
            
        Returns:
            Tuple of (model_weights, num_samples)
        """
        try:
            # Initialize Orchestra components if not done
            if not hasattr(self, 'backbone'):
                self._initialize_orchestra_components()
            
            # Apply global weights if available
            if weights is not None:
                self.set_weights(weights)
            
            # Initialize data iterators if not already done
            refresh_data = kwargs.get("refresh_data", False)
            if refresh_data or not hasattr(self, 'train_iter'):
                self._reset_data_iter()
            
            # Reset memory at the beginning of training
            if self.round_count == 0 and hasattr(self, 'train_dataloader'):
                self._reset_memory(self.train_dataloader)
            
            # Perform local training steps
            num_sample = 0
            total_loss_sum = 0.0
            cluster_loss_sum = 0.0
            deg_loss_sum = 0.0
            
            for step in range(train_steps):
                # Get batch data
                try:
                    x_batch, y_batch, s_w = self.next_batch()
                except Exception as e:
                    self.logger.error(f"Failed to get batch data: {e}")
                    # Fallback: try to get data from kwargs
                    if 'x' in kwargs:
                        x_batch = kwargs['x']
                        y_batch = kwargs.get('y', None)
                        s_w = kwargs.get('s_w', None)
                        self.logger.info("Using data from kwargs as fallback")
                    else:
                        raise ValueError("No input data 'x' found in kwargs and next_batch() failed")
                num_sample += x_batch.shape[0]
                
                # Ensure data is on the correct device
                device = self.exe_device if hasattr(self, 'exe_device') else torch.device('cpu')
                x_batch = x_batch.to(device)
                if y_batch is not None:
                    y_batch = y_batch.to(device)
                
                # Prepare data - expect two augmented views
                if isinstance(x_batch, (list, tuple)) and len(x_batch) >= 2:
                    x1, x2 = x_batch[0].to(device), x_batch[1].to(device)
                else:
                    # If single view provided, create a second view (simple augmentation)
                    x1 = x_batch.to(device)
                    x2 = x_batch + 0.1 * torch.randn_like(x_batch)  # Add noise as simple augmentation
                
                # Generate rotation data for degeneracy regularization
                x3, rotation_labels = self._generate_rotation_data(x1)
                rotation_labels = rotation_labels.to(device)
                
                batch_size = x1.shape[0]
                
                # Get global centroids and ensure they're on the correct device
                C = self.centroids.weight.data.T.to(device)  # [D, N_global]
                
                # Online model forward pass
                z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
                z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)
                
                # Target model forward pass
                with torch.no_grad():
                    self._update_target_model()
                    target_z1 = F.normalize(self.target_projector(self.target_backbone(x1)), dim=1)
                
                # Compute clustering loss 
                cZ2 = z2 @ C  # [N, N_global]
                logpZ2 = torch.log(F.softmax(cZ2 / self.temperature, dim=1))
                
                with torch.no_grad():
                    cP1 = target_z1 @ C
                    tP1 = F.softmax(cP1 / self.temperature, dim=1)
                
                cluster_loss = -torch.sum(tP1 * logpZ2, dim=1).mean()
                
               
                del cZ2, logpZ2, cP1, tP1
                
                # Compute degeneracy regularization loss (rotation prediction)
                deg_preds = self.deg_layer(self.projector(self.backbone(x3)))
                deg_loss = F.cross_entropy(deg_preds, rotation_labels)
                
                # Total loss
                total_loss = self.cluster_weight * cluster_loss + 0.1 * deg_loss
                
                # Clean up temporary variables
                del deg_preds
                
                # Backward pass
                if hasattr(self.model, 'optimizer'):
                    self.model.optimizer.zero_grad()
                    total_loss.backward()
                    self.model.optimizer.step()
                elif hasattr(self.model, 'optimizers'):
                    optimizer = self.model.optimizers()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                else:
                    # Fallback: use automatic optimization if available
                    if hasattr(self.model, 'automatic_optimization') and self.model.automatic_optimization:
                        self.model.backward_step(total_loss)
                    else:
                        raise AttributeError("No optimizer found in model")
                
                # Update memory with target features 
                if step % 2 == 0:  
                    with torch.no_grad():
                        self._update_memory(target_z1.clone())
                
                # Accumulate losses
                total_loss_sum += total_loss.item() * batch_size
                cluster_loss_sum += cluster_loss.item() * batch_size
                deg_loss_sum += deg_loss.item() * batch_size
                
                # Clean up temporary variables
                del x1, x2, x3, rotation_labels, z1, z2, target_z1
                del cluster_loss, deg_loss, total_loss, C
            
            # Perform local clustering at the end of training round
            if self.round_count % 3 == 0:  
                self._local_clustering(device)
            
           
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Calculate average losses
            avg_total_loss = total_loss_sum / num_sample if num_sample > 0 else 0.0
            avg_cluster_loss = cluster_loss_sum / num_sample if num_sample > 0 else 0.0
            avg_deg_loss = deg_loss_sum / num_sample if num_sample > 0 else 0.0
            
            # Log training information
            self.logger.info(
                f"Round {self.round_count}: Avg Total Loss: {avg_total_loss:.4f}, "
                f"Avg Cluster Loss: {avg_cluster_loss:.4f}, "
                f"Avg Deg Loss: {avg_deg_loss:.4f}, "
                f"Samples: {num_sample}"
            )
            
            # Get model weights and sample count for SecretFlow compatibility
            model_weights = self.get_weights()
            
            # Log training information
            logs = {
                'train-loss': avg_total_loss,
                'cluster_loss': avg_cluster_loss,
                'deg_loss': avg_deg_loss
            }
            self.logs = self.transform_metrics(logs)
            self.wrapped_metrics.extend(self.wrap_local_metrics())
            self.epoch_logs = copy.deepcopy(self.logs)
            
            # Return in the format expected by SecretFlow: (weights, num_samples)
            return model_weights, num_sample
            
        except Exception as e:
            self.logger.error(f"Error in train_step: {e}")
            raise
        
        finally:
            self.round_count += 1
    
    def apply_weights(self, weights):
        """
        Apply global weights to local model
        
        Args:
            weights: Global weights to apply
        """
        if isinstance(weights, dict):
            # Handle dictionary format
            if 'centroids' in weights:
                self.centroids.load_state_dict(weights['centroids'])
        else:
            # Handle list format (compatible with SecretFlow)
            self.set_weights(weights)
    
    def get_weights(self):
        """
        Get model weights including Orchestra-specific components
        
        Returns:
            List of model weights
        """
        weights = []
        
        # Add backbone weights
        for param in self.backbone.parameters():
            weights.append(param.data.cpu().numpy())
        
        # Add projector weights
        for param in self.projector.parameters():
            weights.append(param.data.cpu().numpy())
        
        # Add degeneracy layer weights
        for param in self.deg_layer.parameters():
            weights.append(param.data.cpu().numpy())
        
        # Add global centroids
        weights.append(self.centroids.weight.data.cpu().numpy())
        
        return weights
    
    def set_weights(self, weights):
        """
        Set model weights including Orchestra-specific components
        
        Args:
            weights: Weights to set
        """
        # Debug: Check the type and content of weights
        self.logger.info(f"set_weights called with weights type: {type(weights)}")
        
        # Handle callable weights (SecretFlow may pass functions)
        if callable(weights):
            self.logger.info("Weights is callable, attempting to call it")
            try:
                weights = weights()
                self.logger.info(f"Called weights function, result type: {type(weights)}")
            except Exception as e:
                self.logger.error(f"Failed to call weights function: {e}")
                return
        
        if isinstance(weights, dict):
            # Handle dictionary format
            if 'backbone' in weights:
                self.backbone.load_state_dict(weights['backbone'])
            if 'projector' in weights:
                self.projector.load_state_dict(weights['projector'])
            if 'deg_layer' in weights:
                self.deg_layer.load_state_dict(weights['deg_layer'])
            if 'centroids' in weights:
                self.centroids.load_state_dict(weights['centroids'])
        elif isinstance(weights, (list, tuple)):
            # Handle list format
            self.logger.info(f"Setting weights from list with {len(weights)} elements")
            idx = 0
            
            # Set backbone weights
            for param in self.backbone.parameters():
                if idx < len(weights):
                    # Copy the numpy array to avoid non-writable tensor warning
                    weight_array = weights[idx].copy() if hasattr(weights[idx], 'copy') else weights[idx]
                    param.data.copy_(torch.from_numpy(weight_array))
                    idx += 1
            
            # Set projector weights
            for param in self.projector.parameters():
                if idx < len(weights):
                    # Copy the numpy array to avoid non-writable tensor warning
                    weight_array = weights[idx].copy() if hasattr(weights[idx], 'copy') else weights[idx]
                    param.data.copy_(torch.from_numpy(weight_array))
                    idx += 1
            
            # Set degeneracy layer weights
            for param in self.deg_layer.parameters():
                if idx < len(weights):
                    # Copy the numpy array to avoid non-writable tensor warning
                    weight_array = weights[idx].copy() if hasattr(weights[idx], 'copy') else weights[idx]
                    param.data.copy_(torch.from_numpy(weight_array))
                    idx += 1
            
            # Set global centroids
            if idx < len(weights):
                # Copy the numpy array to avoid non-writable tensor warning
                weight_array = weights[idx].copy() if hasattr(weights[idx], 'copy') else weights[idx]
                self.centroids.weight.data.copy_(torch.from_numpy(weight_array))
        else:
            # Handle unexpected types
            self.logger.warning(f"Unexpected weights type: {type(weights)}, attempting to use model.update_weights")
            if hasattr(self.model, 'update_weights'):
                self.model.update_weights(weights)
            else:
                raise TypeError(f"Cannot handle weights of type {type(weights)}")
    
    def get_orchestra_weights(self):
        """
        Get Orchestra-specific weights (centroids, local centroids, memory)
        
        Returns:
            Dictionary containing Orchestra-specific weights
        """
        return {
            'centroids': self.centroids.state_dict(),
            'local_centroids': self.local_centroids.state_dict(),
            'mem_projections': self.mem_projections.state_dict()
        }
    
    def set_orchestra_weights(self, weights):
        """
        Set Orchestra-specific weights
        
        Args:
            weights: Dictionary containing Orchestra-specific weights
        """
        if 'centroids' in weights:
            self.centroids.load_state_dict(weights['centroids'])
        if 'local_centroids' in weights:
            self.local_centroids.load_state_dict(weights['local_centroids'])
        if 'mem_projections' in weights:
            self.mem_projections.load_state_dict(weights['mem_projections'])
    
    def get_cluster_assignments(self, x=None):
        """
        Get cluster assignments for given data
        
        Args:
            x: Input data (if None, uses current memory)
            
        Returns:
            Cluster assignments
        """
        with torch.no_grad():
            if x is not None:
                features = self.backbone(x)
                z = F.normalize(self.projector(features), dim=1)
            else:
                z = self.mem_projections.weight.data.T
            
            # Compute similarities to global centroids
            similarities = z @ self.centroids.weight.data
            assignments = torch.argmax(similarities, dim=1)
            
            return assignments.cpu().numpy()
    
    def extract_features(self, x, feature_type='projection'):
        """
        Extract features from input data using Orchestra components
        
        Args:
            x: Input data tensor
            feature_type: Type of features to extract
                - 'backbone': Raw backbone features
                - 'projection': Normalized projection features (default)
                - 'both': Both backbone and projection features
        
        Returns:
            Extracted features as numpy array or tuple of arrays
        """
        # Set models to evaluation mode
        self.backbone.eval()
        self.projector.eval()
        
        with torch.no_grad():
            # Ensure input is on correct device
            device = self.exe_device if hasattr(self, 'exe_device') else torch.device('cpu')
            if isinstance(x, np.ndarray):
                # Make a writable copy to avoid PyTorch warning
                x = torch.from_numpy(x.copy()).float()
            x = x.to(device)
            
            # Extract backbone features
            backbone_features = self.backbone(x)
            
            if feature_type == 'backbone':
                return backbone_features.cpu().numpy()
            
            # Extract projection features
            projection_features = F.normalize(self.projector(backbone_features), dim=1)
            
            if feature_type == 'projection':
                return projection_features.cpu().numpy()
            elif feature_type == 'both':
                return (
                    backbone_features.cpu().numpy(),
                    projection_features.cpu().numpy()
                )
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}. Use 'backbone', 'projection', or 'both'")
    
    def extract_target_features(self, x, feature_type='projection'):
        """
        Extract features using target (EMA) model
        
        Args:
            x: Input data tensor
            feature_type: Type of features to extract
                - 'backbone': Raw target backbone features
                - 'projection': Normalized target projection features (default)
                - 'both': Both target backbone and projection features
        
        Returns:
            Extracted target features as numpy array or tuple of arrays
        """
        # Set target models to evaluation mode
        self.target_backbone.eval()
        self.target_projector.eval()
        
        with torch.no_grad():
            # Ensure input is on correct device
            device = self.exe_device if hasattr(self, 'exe_device') else torch.device('cpu')
            if isinstance(x, np.ndarray):
                # Make a writable copy to avoid PyTorch warning
                x = torch.from_numpy(x.copy()).float()
            x = x.to(device)
            
            # Extract target backbone features
            target_backbone_features = self.target_backbone(x)
            
            if feature_type == 'backbone':
                return target_backbone_features.cpu().numpy()
            
            # Extract target projection features
            target_projection_features = F.normalize(self.target_projector(target_backbone_features), dim=1)
            
            if feature_type == 'projection':
                return target_projection_features.cpu().numpy()
            elif feature_type == 'both':
                return (
                    target_backbone_features.cpu().numpy(),
                    target_projection_features.cpu().numpy()
                )
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}. Use 'backbone', 'projection', or 'both'")


# PYU wrapper classes for SecretFlow compatibility
class PYUOrchestraStrategy:
    """
    PYU wrapper for Orchestra strategy
    """
    def __init__(self, *args, **kwargs):
        self.strategy = OrchestraStrategy(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.strategy, name)


class PYUOrchestraSimpleStrategy(PYUOrchestraStrategy):
    """
    Simplified PYU wrapper for Orchestra strategy with default parameters
    """
    def __init__(self, model, **kwargs):
        # Set default Orchestra parameters
        default_params = {
            'temperature': 0.1,
            'cluster_weight': 1.0,
            'contrastive_weight': 1.0,
            'deg_weight': 0.1,
            'ema_decay': 0.999,
            'num_local_clusters': 20,
            'num_global_clusters': 10,
            'memory_size': 1024,
            'projection_dim': 512,
            'hidden_dim': 512,
            'epsilon': 0.05,
            'sinkhorn_iterations': 3
        }
        default_params.update(kwargs)
        super().__init__(model, **default_params)