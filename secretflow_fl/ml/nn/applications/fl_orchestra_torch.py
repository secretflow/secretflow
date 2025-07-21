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

"""
Orchestra Federated Learning Application

This module implements the Orchestra federated learning strategy following the original paper:
"Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering"

Key features:
- Unsupervised federated learning
- Global consistency clustering
- Contrastive learning
- EMA target model updates
- Sinkhorn-Knopp equal-size clustering
- Rotation prediction for degeneracy regularization
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from secretflow.device import PYU, reveal
from secretflow.data.ndarray import FedNdarray
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy
from secretflow_fl.ml.nn.fl.fl_model import FLModel


@register_strategy(strategy_name='orchestra', backend='torch')
class OrchestraFLModel(BaseTorchModel):
    """
    Orchestra Federated Learning Model
    
    Implementation of Orchestra strategy for unsupervised federated learning
    following the original paper architecture.
    
    Key Components:
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
        **kwargs: Additional arguments
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
        """
        Initialize Orchestra Federated Learning Model
        
        Args:
            builder_base: Model builder for backbone network
            num_classes: Number of output classes
            temperature: Temperature parameter for contrastive learning
            cluster_weight: Weight for clustering loss
            contrastive_weight: Weight for contrastive loss
            deg_weight: Weight for degeneracy regularization loss
            ema_decay: EMA decay rate for target model updates
            num_local_clusters: Number of local clusters (N_local)
            num_global_clusters: Number of global clusters (N_global)
            memory_size: Size of projection memory bank
            projection_dim: Dimension of projection space
            hidden_dim: Hidden dimension for projection MLP
            epsilon: Epsilon parameter for Sinkhorn-Knopp algorithm
            sinkhorn_iterations: Number of Sinkhorn-Knopp iterations
            **kwargs: Additional arguments
        """
        
        # Orchestra-specific parameters following the original paper
        self.temperature = temperature
        self.cluster_weight = cluster_weight
        self.contrastive_weight = contrastive_weight
        self.deg_weight = deg_weight
        self.ema_decay = ema_decay
        self.num_local_clusters = num_local_clusters  # N_local
        self.num_global_clusters = num_global_clusters  # N_global
        self.memory_size = memory_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Call parent class initialization
        super().__init__(
            builder_base=builder_base,
            num_classes=num_classes,
            **kwargs
        )
        
        self.logger.info("Orchestra FL Model initialized successfully")
    

    
    def fit(
        self,
        x: Union[FedNdarray, Dict[str, FedNdarray]],
        y: Optional[FedNdarray] = None,
        batch_size: Union[int, Dict] = 32,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: Optional[List] = None,
        validation_data: Optional[Tuple] = None,
        shuffle: bool = False,
        class_weight: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_freq: int = 1,
        aggregate_freq: int = 1,
        label_decoder: Optional[callable] = None,
        max_batch_size: int = 20000,
        prefetch_buffer_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        dataset_builder: Optional[callable] = None,
        wait_steps: int = 100,
        **kwargs
    ):
        
        self.logger.info(f"开始Orchestra联邦学习训练，轮数: {epochs}，批次大小: {batch_size}")
        
        # 调用父类的fit方法
        history = super().fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            validation_freq=validation_freq,
            aggregate_freq=aggregate_freq,
            label_decoder=label_decoder,
            max_batch_size=max_batch_size,
            prefetch_buffer_size=prefetch_buffer_size,
            **kwargs
        )
        
        self.logger.info("Orchestra联邦学习训练完成")
        return history
    
    def evaluate(
        self,
        x: Union[FedNdarray, Dict[str, FedNdarray]],
        y: Optional[FedNdarray] = None,
        batch_size: Optional[int] = None,
        verbose: int = 1,
        sample_weight: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
        callbacks: Optional[List] = None,
        max_batch_size: int = 20000,
        dataset_builder: Optional[callable] = None,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        
        self.logger.info("开始Orchestra模型评估")
        
        results = super().evaluate(
            x=x,
            y=y,
            batch_size=batch_size,
            sample_weight=sample_weight,
            dataset_builder=dataset_builder,
            random_seed=random_seed,
            **kwargs
        )
        
        self.logger.info("Orchestra模型评估完成")
        return results
    
    def predict(
        self,
        x: Union[FedNdarray, Dict[str, FedNdarray]],
        batch_size: Optional[int] = None,
        verbose: int = 0,
        steps: Optional[int] = None,
        callbacks: Optional[List] = None,
        max_batch_size: int = 20000,
        dataset_builder: Optional[callable] = None,
        **kwargs
    ):
        
        self.logger.info("开始Orchestra模型预测")
        
        predictions = super().predict(
            x=x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_batch_size=max_batch_size,
            dataset_builder=dataset_builder,
            **kwargs
        )
        
        self.logger.info("Orchestra模型预测完成")
        return predictions
    
    def get_cluster_assignments(self, x: Union[FedNdarray, Dict[str, FedNdarray]], **kwargs):
        
        self.logger.info("获取聚类分配")
        
        # 调用各设备上的聚类分配方法
        assignments = {}
        for device in self.device_list:
            device_assignments = device(lambda model: model.get_cluster_assignments())(
                self.device_y[device]
            )
            assignments[device] = reveal(device_assignments)
        
        return assignments
    
    def save_model(
        self,
        model_path: str,
        is_test: bool = False,
        **kwargs
    ):

        
        self.logger.info(f"保存Orchestra模型到: {model_path}")
        super().save_model(model_path, is_test, **kwargs)
    
    def load_model(
        self,
        model_path: str,
        is_test: bool = False,
        **kwargs
    ):
        
        self.logger.info(f"从{model_path}加载Orchestra模型")
        super().load_model(model_path, is_test, **kwargs)


def create_orchestra_model(
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
    **orchestra_kwargs
) -> OrchestraFLModel:
 
    if not (0.01 <= temperature <= 1.0):
        raise ValueError(f"temperature应在[0.01, 1.0]范围内，当前值: {temperature}")
    
    if not (0.9 <= ema_decay <= 0.999):
        raise ValueError(f"ema_decay应在[0.9, 0.999]范围内，当前值: {ema_decay}")
    
    if cluster_weight < 0 or contrastive_weight < 0 or deg_weight < 0:
        raise ValueError("所有权重参数必须非负")
    
    if num_local_clusters <= 0 or num_global_clusters <= 0:
        raise ValueError("聚类数量必须为正数")
    
    if memory_size <= 0 or projection_dim <= 0 or hidden_dim <= 0:
        raise ValueError("维度参数必须为正数")
    
    if epsilon <= 0 or sinkhorn_iterations <= 0:
        raise ValueError("Sinkhorn-Knopp参数必须为正数")
    
    if num_global_clusters > num_local_clusters:
        import warnings
        warnings.warn(f"全局聚类数({num_global_clusters})大于本地聚类数({num_local_clusters})，这可能不合理")
    
    orchestra_params = {
        'temperature': temperature,
        'cluster_weight': cluster_weight,
        'contrastive_weight': contrastive_weight,
        'deg_weight': deg_weight,
        'ema_decay': ema_decay,
        'num_local_clusters': num_local_clusters,
        'num_global_clusters': num_global_clusters,
        'memory_size': memory_size,
        'projection_dim': projection_dim,
        'hidden_dim': hidden_dim,
        'epsilon': epsilon,
        'sinkhorn_iterations': sinkhorn_iterations,
    }
    
    orchestra_params.update(orchestra_kwargs)
    
    try:
        orchestra_model = OrchestraFLModel(
            builder_base=builder_base,
            num_classes=num_classes,
            **orchestra_params
        )
        
        return orchestra_model
        
    except Exception as e:
        raise RuntimeError(f"创建Orchestra模型失败: {e}") from e