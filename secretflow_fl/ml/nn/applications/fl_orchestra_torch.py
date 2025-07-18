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


class OrchestraFLModel(FLModel):
    """
    Orchestra联邦学习模型应用
    
    实现无监督联邦学习，支持：
    1. 全局一致性聚类
    2. 对比学习
    3. EMA目标模型
    4. Sinkhorn-Knopp等大小聚类
    5. 旋转预测抗退化机制
    """
    
    def __init__(
        self,
        server=None,
        device_list=None,
        model=None,
        aggregator=None,
        strategy="fed_avg_w",
        consensus_num=1,
        backend="torch",
        random_seed=None,
        skip_bn=False,
        **kwargs
    ):
        """
        初始化Orchestra联邦学习模型
        
        Args:
            server: 服务器设备
            device_list: 设备列表
            model: 模型
            aggregator: 聚合器
            strategy: 策略
            consensus_num: 共识数量
            backend: 后端
            random_seed: 随机种子
            skip_bn: 是否跳过批归一化层
            **kwargs: 其他参数
        """
        
        # Orchestra特定参数
        self.temperature = kwargs.get('temperature', 0.1)
        self.cluster_weight = kwargs.get('cluster_weight', 1.0)
        self.contrastive_weight = kwargs.get('contrastive_weight', 1.0)
        self.deg_weight = kwargs.get('deg_weight', 1.0)
        self.num_local_clusters = kwargs.get('num_local_clusters', 16)
        self.num_global_clusters = kwargs.get('num_global_clusters', 128)
        self.memory_size = kwargs.get('memory_size', 128)
        self.ema_decay = kwargs.get('ema_decay', 0.996)
        
        super().__init__(
            server=server,
            device_list=device_list,
            model=model,
            aggregator=aggregator,
            strategy=strategy,
            consensus_num=consensus_num,
            backend=backend,
            random_seed=random_seed,
            skip_bn=skip_bn,
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Orchestra FL模型初始化完成")
    

    
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
        """
        训练Orchestra联邦学习模型
        
        Args:
            x: 训练数据（无监督学习中不需要标签）
            y: 标签数据（Orchestra中可选）
            batch_size: 批次大小
            epochs: 训练轮数
            verbose: 详细程度
            callbacks: 回调函数列表
            validation_data: 验证数据
            shuffle: 是否打乱数据
            class_weight: 类别权重
            sample_weight: 样本权重
            validation_freq: 验证频率
            aggregate_freq: 聚合频率
            label_decoder: 标签解码器
            max_batch_size: 最大批次大小
            prefetch_buffer_size: 预取缓冲区大小
            random_seed: 随机种子
            dataset_builder: 数据集构建器
            wait_steps: 等待步骤
            **kwargs: 其他参数
        """
        
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
        """
        评估Orchestra模型
        
        Args:
            x: 测试数据
            y: 测试标签
            batch_size: 批次大小
            verbose: 详细程度
            sample_weight: 样本权重
            steps: 评估步数
            callbacks: 回调函数
            max_batch_size: 最大批次大小
            dataset_builder: 数据集构建器
            random_seed: 随机种子
            **kwargs: 其他参数
        
        Returns:
            评估结果
        """
        
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
        """
        使用Orchestra模型进行预测
        
        Args:
            x: 输入数据
            batch_size: 批次大小
            verbose: 详细程度
            steps: 预测步数
            callbacks: 回调函数
            max_batch_size: 最大批次大小
            dataset_builder: 数据集构建器
            **kwargs: 其他参数
        
        Returns:
            预测结果
        """
        
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
        """
        获取数据的聚类分配
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
        
        Returns:
            聚类分配结果
        """
        
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
        """
        保存Orchestra模型
        
        Args:
            model_path: 模型保存路径
            is_test: 是否为测试模式
            **kwargs: 其他参数
        """
        
        self.logger.info(f"保存Orchestra模型到: {model_path}")
        super().save_model(model_path, is_test, **kwargs)
    
    def load_model(
        self,
        model_path: str,
        is_test: bool = False,
        **kwargs
    ):
        """
        加载Orchestra模型
        
        Args:
            model_path: 模型路径
            is_test: 是否为测试模式
            **kwargs: 其他参数
        """
        
        self.logger.info(f"从{model_path}加载Orchestra模型")
        super().load_model(model_path, is_test, **kwargs)


def create_orchestra_model(
    model,
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
    """
    创建Orchestra联邦学习模型的便捷函数 - 增强参数验证
    
    Args:
        model: 基础模型（backbone）
        num_classes: 类别数量
        temperature: 对比学习温度参数 (0.01-1.0)
        cluster_weight: 聚类损失权重 (>= 0)
        contrastive_weight: 对比学习损失权重 (>= 0)
        deg_weight: 抗退化损失权重 (>= 0)
        ema_decay: EMA衰减率 (0.9-0.999)
        num_local_clusters: 本地聚类数量 (> 0)
        num_global_clusters: 全局聚类数量 (> 0)
        memory_size: 投影内存大小 (> 0)
        projection_dim: 投影维度 (> 0)
        hidden_dim: 隐藏层维度 (> 0)
        epsilon: Sinkhorn-Knopp算法的epsilon参数 (> 0)
        sinkhorn_iterations: Sinkhorn-Knopp迭代次数 (> 0)
        **orchestra_kwargs: Orchestra策略的其他参数
    
    Returns:
        OrchestraFLModel实例
        
    Raises:
        ValueError: 当参数不在有效范围内时
    """
    # 参数验证
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
    
    # 合理性检查
    if num_global_clusters > num_local_clusters:
        import warnings
        warnings.warn(f"全局聚类数({num_global_clusters})大于本地聚类数({num_local_clusters})，这可能不合理")
    
    # 设置Orchestra参数
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
    
    # 合并用户提供的参数
    orchestra_params.update(orchestra_kwargs)
    
    try:
        # 创建Orchestra模型
        orchestra_model = OrchestraFLModel(
            model=model,
            num_classes=num_classes,
            **orchestra_params
        )
        
        return orchestra_model
        
    except Exception as e:
        raise RuntimeError(f"创建Orchestra模型失败: {e}") from e