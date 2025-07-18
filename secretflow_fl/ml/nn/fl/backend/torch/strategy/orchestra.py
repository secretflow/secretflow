import copy
from typing import Tuple, List, Optional
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


def sknopp(cZ, lamd=25, max_iters=100):
    """
    Sinkhorn-Knopp算法实现，用于等大小聚类 - 参考原始论文增强数值稳定性
    
    Args:
        cZ: 聚类分配矩阵 [N_samples, N_centroids]
        lamd: 温度参数
        max_iters: 最大迭代次数
    Returns:
        等大小的聚类分配概率矩阵
    """
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape
        
        # 数值稳定性：减去最大值避免溢出
        cZ_stable = cZ - cZ.max(dim=1, keepdim=True)[0]
        probs = F.softmax(cZ_stable * lamd, dim=1).T  # [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device, dtype=probs.dtype) / N_centroids
        c = torch.ones((N_samples, 1), device=probs.device, dtype=probs.dtype) / N_samples

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            # 添加数值稳定性检查
            probs_c = probs @ c
            probs_c = torch.clamp(probs_c, min=1e-8)  # 避免除零
            r = inv_N_centroids / probs_c
            
            r_probs = r.T @ probs
            r_probs = torch.clamp(r_probs, min=1e-8)  # 避免除零
            c_new = inv_N_samples / r_probs.T
            
            # 检查收敛性
            if it % 10 == 0:
                c_ratio = c / torch.clamp(c_new, min=1e-8)
                err = torch.nansum(torch.abs(c_ratio - 1))
                
            c = c_new
            
            # 提前停止条件
            if err < 1e-2:
                break
                
            # 额外的数值稳定性：重新归一化
            if it % 20 == 0:
                r = r / torch.clamp(r.sum(), min=1e-8) * N_centroids
                c = c / torch.clamp(c.sum(), min=1e-8) * N_samples

        # 返回转置后的概率矩阵 [N_samples, N_centroids]
        result = (r @ c.T * probs).T * N_samples
        
        # 最终归一化确保概率分布
        result = result / torch.clamp(result.sum(dim=1, keepdim=True), min=1e-8)
        
        return result


def sinkhorn_knopp(Q, num_iters=3, epsilon=0.05):
    """Sinkhorn-Knopp算法实现 - 参考原始论文增强数值稳定性"""
    with torch.no_grad():
        # 数值稳定性：减去最大值避免溢出
        Q = Q - Q.max(dim=1, keepdim=True)[0]
        Q = torch.exp(Q / epsilon)
        
        B, K = Q.shape
        
        # 初始化行和列的权重
        r = torch.ones(B, 1, device=Q.device, dtype=Q.dtype) / B
        c = torch.ones(1, K, device=Q.device, dtype=Q.dtype) / K
        
        # 迭代优化
        for iteration in range(num_iters):
            # 行归一化：确保每行和为1/B
            row_sums = Q.sum(dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-8)  # 避免除零
            Q = Q * r / row_sums
            
            # 列归一化：确保每列和为1/K
            col_sums = Q.sum(dim=0, keepdim=True)
            col_sums = torch.clamp(col_sums, min=1e-8)  # 避免除零
            Q = Q * c / col_sums
            
            # 检查收敛性（可选）
            if iteration > 0 and iteration % 2 == 0:
                row_diff = torch.abs(Q.sum(dim=1) - 1.0/B).max()
                col_diff = torch.abs(Q.sum(dim=0) - 1.0/K).max()
                if row_diff < 1e-6 and col_diff < 1e-6:
                    break
        
        # 最终归一化确保概率分布
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        return Q


class ProjectionMLP(nn.Module):
    """Orchestra投影网络 - 参考原始论文实现"""
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(ProjectionMLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 使用BatchNorm1d以保持与原始论文一致，但在联邦学习中可能需要调整
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        # 第二层不使用affine变换，与原始论文保持一致
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)
        
        # 备用LayerNorm，在BatchNorm不稳定时使用
        self.use_layernorm = False
        self.layer1_ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.layer2_ln = nn.LayerNorm(out_dim, elementwise_affine=False)

    def forward(self, x):
        if self.use_layernorm or x.shape[0] == 1:  # 批次大小为1时使用LayerNorm
            x = F.relu(self.layer1_ln(self.layer1(x)))
            x = self.layer2_ln(self.layer2(x))
        else:
            x = F.relu(self.layer1_bn(self.layer1(x)))
            x = self.layer2_bn(self.layer2(x))
        return x
    
    def set_layernorm_mode(self, use_layernorm=True):
        """切换到LayerNorm模式以提高联邦学习稳定性"""
        self.use_layernorm = use_layernorm


class OrchestraStrategy(BaseTorchModel):
    """
    Orchestra联邦学习策略：实现无监督聚类的联邦学习
    
    核心特性：
    1. 无监督学习：不需要标签数据
    2. 全局一致性聚类：确保各客户端聚类结果一致
    3. 对比学习：通过对比损失学习特征表示
    4. 聚类损失：通过聚类中心学习数据分布
    5. EMA目标模型：指数移动平均更新目标网络
    6. Sinkhorn-Knopp等大小聚类
    7. 旋转预测任务作为抗退化机制
    """
    
    def __init__(self, builder_base, **kwargs):
        # 正确传递builder_base参数给BaseTorchModel
        logging.warning(f"OrchestraStrategy.__init__ called. builder_base type: {type(builder_base)}, kwargs keys: {list(kwargs.keys())}")
        super().__init__(builder_base=builder_base, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Orchestra特定参数
        self.temperature = kwargs.get('temperature', 0.1)  # 主要温度参数
        self.cluster_weight = kwargs.get('cluster_weight', 1.0)  # 聚类损失权重
        self.contrastive_weight = kwargs.get('contrastive_weight', 1.0)  # 对比损失权重
        self.deg_weight = kwargs.get('deg_weight', 1.0)  # 抗退化损失权重
        
        # 聚类参数
        self.num_local_clusters = kwargs.get('num_local_clusters', 50)  # 本地聚类数量
        self.num_global_clusters = kwargs.get('num_global_clusters', 10)  # 全局聚类数量
        self.memory_size = kwargs.get('memory_size', 1024)  # 投影内存大小
        
        # EMA参数
        self.ema_value = kwargs.get('ema_value', 0.99)  # EMA衰减率
        
        # 全局聚类中心（由服务器维护）
        self.global_cluster_centers = None
        self.local_cluster_centers = None
        
        # 投影内存
        self.projection_memory = None
        
        # 目标模型（EMA）
        self.target_model = None
        self.target_projector = None
        
        # 投影网络和抗退化层
        self.projector = None
        self.deg_layer = None  # 旋转预测层
        
        # 轮次计数
        self.rounds_done = 0
        
        # 初始化标志
        self._initialized = False
        
    def _initialize_orchestra_components(self):
        """初始化Orchestra特定组件"""
        if self._initialized or self.model is None:
            return
            
        device = next(self.model.parameters()).device
        
        # 动态获取backbone输出维度
        self.model.eval()
        with torch.no_grad():
            # 尝试不同的输入尺寸来检测模型输出维度
            backbone_dim = None
            test_sizes = [(1, 3, 32, 32), (1, 3, 224, 224), (1, 784), (1, 128)]  # 支持多种输入格式
            
            for test_size in test_sizes:
                try:
                    test_input = torch.randn(*test_size).to(device)
                    test_output = self.model(test_input)
                    if isinstance(test_output, tuple):
                        test_output = test_output[0]
                    backbone_dim = test_output.shape[-1]
                    self.logger.info(f"成功检测到模型输出维度: {backbone_dim}，输入尺寸: {test_size}")
                    break
                except Exception as e:
                    continue
            
            if backbone_dim is None:
                self.logger.warning("无法动态检测模型输出维度，使用默认值512")
                backbone_dim = 512  # 使用更通用的默认值
        
        self.logger.info(f"检测到模型输出维度: {backbone_dim}")
        
        # 初始化投影网络
        self.projector = ProjectionMLP(backbone_dim, hidden_dim=512, out_dim=512).to(device)
        
        # 初始化目标模型（EMA）
        self.target_model = copy.deepcopy(self.model)
        self.target_projector = copy.deepcopy(self.projector)
        
        # 冻结目标模型参数
        for param in self.target_model.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
        # 初始化抗退化层（旋转预测）
        self.deg_layer = nn.Linear(512, 4).to(device)  # 4个旋转角度：0, 90, 180, 270
        
        # 初始化聚类中心
        self.global_cluster_centers = F.normalize(
            torch.randn(self.num_global_clusters, 512, device=device),
            dim=1
        )
        self.local_cluster_centers = F.normalize(
            torch.randn(self.num_local_clusters, 512, device=device),
            dim=1
        )
        
        # 初始化投影内存
        self.projection_memory = nn.Linear(self.memory_size, 512, bias=False).to(device)
        
        self._initialized = True
        self.logger.info("Orchestra组件初始化完成")
        
    @torch.no_grad()
    def _update_target_model(self):
        """使用EMA更新目标模型"""
        if self.target_model is None or self.target_projector is None:
            return
            
        tau = self.ema_value
        
        # 更新目标backbone
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
            
        # 更新目标投影网络
        for target_param, online_param in zip(self.target_projector.parameters(), self.projector.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
             
    @torch.no_grad()
    def _reset_memory(self, data_loader):
        """重置投影内存"""
        if self.target_model is None or self.target_projector is None:
            return
            
        self.target_model.eval()
        self.target_projector.eval()
        
        # 生成特征库
        proj_bank = []
        n_samples = 0
        
        for batch in data_loader:
            if n_samples >= self.memory_size:
                break
                
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            if hasattr(x, 'to'):
                x = x.to(next(self.target_model.parameters()).device)
                
            # 获取目标模型的投影
            features = self.target_model(x)
            if isinstance(features, tuple):
                features = features[0]
            projections = F.normalize(self.target_projector(features), dim=1)
            
            proj_bank.append(projections)
            n_samples += x.shape[0]
            
        if proj_bank:
            proj_bank = torch.cat(proj_bank, dim=0).contiguous()
            if n_samples > self.memory_size:
                proj_bank = proj_bank[:self.memory_size]
                
            # 保存投影到内存
            self.projection_memory.weight.data.copy_(proj_bank.T)
            
    @torch.no_grad()
    def _update_memory(self, new_projections):
        """更新投影内存"""
        if self.projection_memory is None:
            return
            
        N = new_projections.shape[0]
        # 移动内存
        self.projection_memory.weight.data[:, :-N] = self.projection_memory.weight.data[:, N:].detach().clone()
        # 添加新投影
        self.projection_memory.weight.data[:, -N:] = new_projections.T.detach().clone()
        
    def _local_clustering(self):
        """本地聚类（使用Sinkhorn-Knopp算法）"""
        if self.projection_memory is None:
            return
            
        with torch.no_grad():
            # 获取投影内存数据 [memory_size, feature_dim]
            Z = self.projection_memory.weight.data.T.detach().clone()
            
            # 随机初始化聚类中心
            centroids = Z[np.random.choice(Z.shape[0], self.num_local_clusters, replace=False)]
            
            # 本地聚类迭代
            local_iters = 5
            for it in range(local_iters):
                # 使用Sinkhorn-Knopp算法进行等大小聚类
                assigns = sknopp(Z @ centroids.T, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                
                # 更新聚类中心
                for index in range(self.num_local_clusters):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    if selected.numel() == 0:
                        selected = torch.randint(len(Z), (1,))
                    elif selected.numel() == 1:
                        selected = selected.unsqueeze(0)
                        
                    selected_features = torch.index_select(Z, 0, selected)
                    if selected_features.shape[0] == 0:
                        selected_features = Z[torch.randint(len(Z), (1,))]
                        
                    centroids[index] = F.normalize(selected_features.mean(dim=0), dim=0)
                    
            # 保存本地聚类中心
            self.local_cluster_centers = centroids.detach().clone()
            
    def _generate_rotation_data(self, x):
        """生成旋转数据用于抗退化"""
        batch_size = x.shape[0]
        angles = torch.randint(0, 4, (batch_size,), device=x.device)
        
        # 检查输入维度，如果不是图像数据，则简化处理
        if len(x.shape) < 4:  # 不是图像数据 (batch, channel, height, width)
            # 对于非图像数据，使用简单的数据增强
            rotated_x = []
            for i, angle in enumerate(angles):
                if angle == 0:
                    rotated_x.append(x[i])
                elif angle == 1:  # 添加噪声
                    noise = torch.randn_like(x[i]) * 0.01
                    rotated_x.append(x[i] + noise)
                elif angle == 2:  # 翻转符号
                    rotated_x.append(-x[i])
                else:  # 缩放
                    rotated_x.append(x[i] * 0.9)
        else:
            # 对于图像数据，使用旋转
            rotated_x = []
            for i, angle in enumerate(angles):
                if angle == 0:
                    rotated_x.append(x[i])
                elif angle == 1:  # 90度
                    rotated_x.append(torch.rot90(x[i], k=1, dims=[-2, -1]))
                elif angle == 2:  # 180度
                    rotated_x.append(torch.rot90(x[i], k=2, dims=[-2, -1]))
                else:  # 270度
                    rotated_x.append(torch.rot90(x[i], k=3, dims=[-2, -1]))
                
        return torch.stack(rotated_x), angles
        
    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Orchestra训练步骤 - 完整实现
        
        Args:
            weights: 全局权重（包含模型参数和聚类中心）
            cur_steps: 当前训练步数
            train_steps: 本地训练步数
            kwargs: 策略特定参数
        Returns:
            训练后的参数和样本数量
        """
        assert self.model is not None, "Model cannot be none"
        
        # 初始化Orchestra组件
        self._initialize_orchestra_components()
        
        self.model.train()
        if self.projector is not None:
            self.projector.train()
        if self.deg_layer is not None:
            self.deg_layer.train()
        
        # 刷新数据迭代器
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
            
        # 应用全局权重
        if weights is not None and len(weights) > 0:
            self.set_weights(weights)
            
        # 在训练开始时重置内存（如果是新轮次）
        if cur_steps == 0 and hasattr(self, 'train_data_loader') and self.train_data_loader is not None:
            self._reset_memory(self.train_data_loader)
            
        num_sample = 0
        total_loss = 0.0
        total_cluster_loss = 0.0
        total_deg_loss = 0.0
        
        self.logger.info(f"开始Orchestra本地训练，训练步数: {train_steps}")
        
        for step in range(train_steps):
            try:
                # 获取数据
                batch_data = self.next_batch()
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    x, y = batch_data[0], batch_data[1]
                else:
                    x = batch_data
                    y = None
                
                # 处理数据格式
                if isinstance(x, (tuple, list)):
                    # Orchestra需要两个增强视图和一个旋转视图
                    if len(x) >= 2:
                        x1, x2 = x[0], x[1]
                    else:
                        x1 = x2 = x[0]
                else:
                    # 如果只有一个视图，复制一份
                    x1 = x2 = x
                
                # 生成旋转数据用于抗退化
                x3, rotation_labels = self._generate_rotation_data(x1)
                
                batch_size = x1.shape[0]
                num_sample += batch_size
                
                # 在线模型前向传播
                z1 = self.model(x1)
                if isinstance(z1, tuple):
                    z1 = z1[0]
                z1_proj = F.normalize(self.projector(z1), dim=1)
                
                z2 = self.model(x2)
                if isinstance(z2, tuple):
                    z2 = z2[0]
                z2_proj = F.normalize(self.projector(z2), dim=1)
                
                # 目标模型前向传播（用于聚类损失）
                with torch.no_grad():
                    self._update_target_model()
                    tz1 = self.target_model(x1)
                    if isinstance(tz1, tuple):
                        tz1 = tz1[0]
                    tz1_proj = F.normalize(self.target_projector(tz1), dim=1)
                
                # 计算聚类损失
                cluster_loss = self._compute_orchestra_cluster_loss(z2_proj, tz1_proj)
                
                # 计算抗退化损失（旋转预测）
                z3 = self.model(x3)
                if isinstance(z3, tuple):
                    z3 = z3[0]
                z3_proj = self.projector(z3)
                deg_preds = self.deg_layer(z3_proj)
                deg_loss = F.cross_entropy(deg_preds, rotation_labels)
                
                # 总损失
                loss = cluster_loss + self.deg_weight * deg_loss
                
                # 反向传播
                if hasattr(self.model, 'optimizer'):
                    self.model.optimizer.zero_grad()
                    
                # 同时优化projector和deg_layer
                if hasattr(self.projector, 'parameters'):
                    proj_optimizer = torch.optim.Adam(self.projector.parameters(), lr=1e-3)
                    proj_optimizer.zero_grad()
                    
                if hasattr(self.deg_layer, 'parameters'):
                    deg_optimizer = torch.optim.Adam(self.deg_layer.parameters(), lr=1e-3)
                    deg_optimizer.zero_grad()
                
                loss.backward()
                
                if hasattr(self.model, 'optimizer'):
                    self.model.optimizer.step()
                if 'proj_optimizer' in locals():
                    proj_optimizer.step()
                if 'deg_optimizer' in locals():
                    deg_optimizer.step()
                
                # 更新投影内存
                with torch.no_grad():
                    self._update_memory(tz1_proj)
                
                total_loss += loss.item()
                total_cluster_loss += cluster_loss.item()
                total_deg_loss += deg_loss.item()
                
            except Exception as e:
                self.logger.error(f"训练步骤 {step} 出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        # 训练结束后进行本地聚类
        self._local_clustering()
        
        # 计算平均损失
        avg_loss = total_loss / train_steps if train_steps > 0 else 0.0
        avg_cluster_loss = total_cluster_loss / train_steps if train_steps > 0 else 0.0
        avg_deg_loss = total_deg_loss / train_steps if train_steps > 0 else 0.0
        
        # 记录训练日志
        logs = {
            "train-loss": avg_loss,
            "cluster-loss": avg_cluster_loss,
            "deg-loss": avg_deg_loss,
            "num_samples": num_sample
        }
        
        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)
        
        self.logger.info(f"Orchestra本地训练完成，平均损失: {avg_loss:.4f}, 样本数: {num_sample}")
        
        # 更新轮次计数
        self.rounds_done += 1
        
        # 返回更新后的权重 - 简化格式以提高稳定性
        # 将所有权重扁平化为numpy数组列表，符合SecretFlow标准格式
        weights_to_return = []
        
        # 1. 模型权重
        for param in self.model.parameters():
            weights_to_return.append(param.data.cpu().numpy())
            
        # 2. 投影网络权重
        if self.projector is not None:
            for param in self.projector.parameters():
                weights_to_return.append(param.data.cpu().numpy())
        
        # 3. 抗退化层权重
        if self.deg_layer is not None:
            for param in self.deg_layer.parameters():
                weights_to_return.append(param.data.cpu().numpy())
                
        # 4. 本地聚类中心（如果存在）
        if self.local_cluster_centers is not None:
            weights_to_return.append(self.local_cluster_centers.cpu().numpy())
            
        self.logger.info(f"返回权重数量: {len(weights_to_return)}，本地聚类中心形状: {self.local_cluster_centers.shape if self.local_cluster_centers is not None else 'None'}")

        return weights_to_return, num_sample
    
    def _compute_cluster_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算聚类损失（兼容性方法）
        
        Args:
            features: 特征表示 [batch_size, feature_dim]
        Returns:
            聚类损失
        """
        if self.global_cluster_centers is None:
            # 如果没有全局聚类中心，使用随机初始化
            feature_dim = features.shape[-1]
            self.global_cluster_centers = torch.randn(
                self.num_global_clusters, feature_dim, 
                device=features.device
            )
        
        # 确保聚类中心在正确的设备上
        if isinstance(self.global_cluster_centers, np.ndarray):
            cluster_centers = torch.from_numpy(self.global_cluster_centers).to(features.device)
        else:
            cluster_centers = self.global_cluster_centers.to(features.device)
        
        # 计算特征到聚类中心的距离
        # features: [batch_size, feature_dim]
        # cluster_centers: [num_clusters, feature_dim]
        distances = torch.cdist(features, cluster_centers)  # [batch_size, num_clusters]
        
        # 软分配：使用softmax计算分配概率
        assignments = F.softmax(-distances / self.temperature, dim=1)
        
        # 聚类损失：最小化特征到最近聚类中心的距离
        min_distances, _ = torch.min(distances, dim=1)
        cluster_loss = torch.mean(min_distances)
        
        return cluster_loss
    
    def _compute_orchestra_cluster_loss(self, z_proj: torch.Tensor, tz_proj: torch.Tensor) -> torch.Tensor:
        """
        计算Orchestra聚类损失
        
        Args:
            z_proj: 在线模型投影 [batch_size, proj_dim]
            tz_proj: 目标模型投影 [batch_size, proj_dim]
        Returns:
            聚类损失
        """
        batch_size = z_proj.shape[0]
        
        # 如果全局聚类中心未初始化，使用目标投影初始化
        if self.global_cluster_centers is None:
            proj_dim = tz_proj.shape[1]
            self.global_cluster_centers = F.normalize(
                torch.randn(self.num_global_clusters, proj_dim, device=tz_proj.device),
                dim=1
            )
            self.logger.info(f"初始化Orchestra全局聚类中心: {self.global_cluster_centers.shape}")
        
        # 计算目标投影到全局聚类中心的相似度
        sim_matrix = torch.mm(tz_proj, self.global_cluster_centers.T.to(tz_proj.dtype)) / self.temperature
        
        # 使用Sinkhorn-Knopp算法获得聚类分配
        with torch.no_grad():
            cluster_assignments = sknopp(sim_matrix.detach(), max_iters=3)
        
        # 计算在线投影到全局聚类中心的相似度
        online_sim = torch.mm(z_proj, self.global_cluster_centers.T) / self.temperature
        
        # 聚类损失：在线投影应该与目标投影的聚类分配一致
        cluster_loss = -torch.mean(torch.sum(cluster_assignments * F.log_softmax(online_sim, dim=1), dim=1))
        
        return cluster_loss
    
    def _compute_contrastive_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            projections: 投影特征 [batch_size, projection_dim]
        Returns:
            对比损失
        """
        batch_size = projections.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=projections.device)
        
        # 归一化投影
        projections = F.normalize(projections, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # 创建正样本掩码（这里简化处理，实际应该基于数据增强）
        # 假设batch中相邻的样本是正样本对
        positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        for i in range(0, batch_size - 1, 2):
            if i + 1 < batch_size:
                positive_mask[i, i + 1] = True
                positive_mask[i + 1, i] = True
        
        # 负样本掩码
        negative_mask = ~positive_mask
        # 移除对角线（自己与自己的相似度）
        negative_mask.fill_diagonal_(False)
        
        # 如果没有正样本对，返回零损失
        if not positive_mask.any():
            return torch.tensor(0.0, device=projections.device)
        
        # 计算对比损失
        positive_similarities = similarity_matrix[positive_mask]
        negative_similarities = similarity_matrix[negative_mask].view(batch_size, -1)
        
        # InfoNCE损失
        logits = torch.cat([positive_similarities.unsqueeze(1), negative_similarities], dim=1)
        labels = torch.zeros(positive_similarities.shape[0], dtype=torch.long, device=projections.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return contrastive_loss
    
    def set_weights(self, weights):
        """设置模型权重 - 改进的权重管理"""
        if not weights:
            self.logger.warning("接收到空权重列表")
            return
            
        try:
            device = next(self.model.parameters()).device
            weight_idx = 0
            
            # 1. 设置模型权重
            model_param_count = len(list(self.model.parameters()))
            if weight_idx + model_param_count <= len(weights):
                for param in self.model.parameters():
                    weight_data = weights[weight_idx]
                    if isinstance(weight_data, np.ndarray):
                        param.data.copy_(torch.from_numpy(weight_data).to(device))
                    elif isinstance(weight_data, torch.Tensor):
                        param.data.copy_(weight_data.to(device))
                    else:
                        param.data.copy_(torch.tensor(weight_data, device=device))
                    weight_idx += 1
                self.logger.debug(f"成功设置模型权重，参数数量: {model_param_count}")
            else:
                self.logger.error(f"权重数量不足以设置模型参数，需要: {model_param_count}, 可用: {len(weights) - weight_idx}")
                return
            
            # 2. 设置投影网络权重
            if self.projector is not None:
                proj_param_count = len(list(self.projector.parameters()))
                if weight_idx + proj_param_count <= len(weights):
                    for param in self.projector.parameters():
                        weight_data = weights[weight_idx]
                        if isinstance(weight_data, np.ndarray):
                            param.data.copy_(torch.from_numpy(weight_data).to(device))
                        elif isinstance(weight_data, torch.Tensor):
                            param.data.copy_(weight_data.to(device))
                        else:
                            param.data.copy_(torch.tensor(weight_data, device=device))
                        weight_idx += 1
                    self.logger.debug(f"成功设置投影网络权重，参数数量: {proj_param_count}")
                else:
                    self.logger.warning(f"权重数量不足以设置投影网络参数，跳过")
            
            # 3. 设置抗退化层权重
            if self.deg_layer is not None:
                deg_param_count = len(list(self.deg_layer.parameters()))
                if weight_idx + deg_param_count <= len(weights):
                    for param in self.deg_layer.parameters():
                        weight_data = weights[weight_idx]
                        if isinstance(weight_data, np.ndarray):
                            param.data.copy_(torch.from_numpy(weight_data).to(device))
                        elif isinstance(weight_data, torch.Tensor):
                            param.data.copy_(weight_data.to(device))
                        else:
                            param.data.copy_(torch.tensor(weight_data, device=device))
                        weight_idx += 1
                    self.logger.debug(f"成功设置抗退化层权重，参数数量: {deg_param_count}")
                else:
                    self.logger.warning(f"权重数量不足以设置抗退化层参数，跳过")
            
            # 4. 设置全局聚类中心（如果存在）
            if weight_idx < len(weights):
                centers_data = weights[weight_idx]
                try:
                    if isinstance(centers_data, np.ndarray):
                        self.global_cluster_centers = torch.from_numpy(centers_data).to(device).float()
                    elif isinstance(centers_data, torch.Tensor):
                        self.global_cluster_centers = centers_data.to(device).float()
                    else:
                        self.global_cluster_centers = torch.tensor(centers_data, device=device, dtype=torch.float32)
                    
                    # 验证聚类中心形状
                    expected_shape = (self.num_global_clusters, self.projector.layer2.out_features if self.projector else 512)
                    if self.global_cluster_centers.shape != expected_shape:
                        self.logger.warning(f"全局聚类中心形状不匹配，期望: {expected_shape}, 实际: {self.global_cluster_centers.shape}")
                        # 尝试重新初始化
                        self.global_cluster_centers = torch.randn(expected_shape, device=device, dtype=torch.float32)
                        self.logger.info("重新初始化全局聚类中心")
                    else:
                        self.logger.debug(f"成功设置全局聚类中心，形状: {self.global_cluster_centers.shape}")
                        
                except Exception as e:
                    self.logger.error(f"设置全局聚类中心时出错: {e}，重新初始化")
                    expected_shape = (self.num_global_clusters, self.projector.layer2.out_features if self.projector else 512)
                    self.global_cluster_centers = torch.randn(expected_shape, device=device, dtype=torch.float32)
                    
        except Exception as e:
            self.logger.error(f"设置权重时出错: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    def apply_weights(self, weights, **kwargs):
        """
        应用全局权重到本地模型
        
        Args:
            weights: 全局权重
            kwargs: 额外参数，可能包含聚类中心
        """
        if weights is not None:
            self.set_weights(weights)
        
        # 更新全局聚类中心
        if 'global_cluster_centers' in kwargs:
            self.global_cluster_centers = kwargs['global_cluster_centers']
            self.logger.info(f"更新全局聚类中心: {self.global_cluster_centers.shape}")
    
    def get_orchestra_weights(self):
        """
        获取Orchestra特定的权重，包括聚类中心和投影内存
        
        Returns:
            包含Orchestra权重的字典
        """
        weights = {}
        
        if self.global_cluster_centers is not None:
            weights['global_cluster_centers'] = self.global_cluster_centers.clone()
        
        if self.local_cluster_centers is not None:
            weights['local_cluster_centers'] = self.local_cluster_centers.clone()
        
        if self.projection_memory is not None:
            if hasattr(self.projection_memory, 'weight'):
                weights['projection_memory'] = self.projection_memory.weight.data.clone()
            else:
                weights['projection_memory'] = self.projection_memory.clone()
        
        return weights
    
    def set_orchestra_weights(self, weights):
        """
        设置Orchestra特定的权重
        
        Args:
            weights: 包含Orchestra权重的字典
        """
        if 'global_cluster_centers' in weights:
            self.global_cluster_centers = weights['global_cluster_centers'].clone()
            
        if 'local_cluster_centers' in weights:
            self.local_cluster_centers = weights['local_cluster_centers'].clone()
            
        if 'projection_memory' in weights and self.projection_memory is not None:
            if hasattr(self.projection_memory, 'weight'):
                self.projection_memory.weight.data.copy_(weights['projection_memory'])
            else:
                self.projection_memory = weights['projection_memory'].clone()
    
    def get_cluster_assignments(self, data_loader=None) -> np.ndarray:
        """
        获取数据的聚类分配
        
        Args:
            data_loader: 数据加载器，如果为None则使用训练数据
        Returns:
            聚类分配数组
        """
        if not self._initialized:
            self.logger.warning("Orchestra组件未初始化，返回空的聚类分配")
            return np.array([])
            
        self.model.eval()
        if self.projector is not None:
            self.projector.eval()
        assignments = []
        
        if data_loader is None:
            data_loader = self.train_data_loader
        
        if data_loader is None:
            self.logger.warning("没有可用的数据加载器，返回空的聚类分配")
            return np.array([])
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                # 获取模型特征
                features = self.model(x)
                if isinstance(features, tuple):
                    features = features[0]
                
                # 使用投影网络获取投影特征（与聚类中心维度一致）
                if self.projector is not None:
                    projected_features = F.normalize(self.projector(features), dim=1)
                else:
                    projected_features = features
                
                # 计算到聚类中心的距离
                if self.global_cluster_centers is not None:
                    if isinstance(self.global_cluster_centers, np.ndarray):
                        cluster_centers = torch.from_numpy(self.global_cluster_centers).to(projected_features.device)
                    else:
                        cluster_centers = self.global_cluster_centers.to(projected_features.device)
                    
                    # 确保维度匹配
                    if projected_features.shape[1] != cluster_centers.shape[1]:
                        self.logger.warning(f"特征维度不匹配: {projected_features.shape[1]} vs {cluster_centers.shape[1]}，跳过此批次")
                        continue
                    
                    distances = torch.cdist(projected_features, cluster_centers)
                    batch_assignments = torch.argmin(distances, dim=1)
                    assignments.append(batch_assignments.cpu().numpy())
        
        return np.concatenate(assignments) if assignments else np.array([])


@register_strategy(strategy_name="orchestra", backend="torch")
class PYUOrchestraStrategy(OrchestraStrategy):
    """
    Orchestra策略的PYU包装类
    用于在SecretFlow的PYU设备上运行
    """
    
    def __init__(self, builder_base, random_seed=None, skip_bn=False, **kwargs):
        # 确保正确传递所有参数给父类
        logging.warning(f"PYUOrchestraStrategy.__init__ called. builder_base type: {type(builder_base)}, kwargs keys: {list(kwargs.keys())}")
        super().__init__(builder_base=builder_base, random_seed=random_seed, skip_bn=skip_bn, **kwargs)


@register_strategy(strategy_name="orchestra_simple", backend="torch")
class PYUOrchestraSimpleStrategy(OrchestraStrategy):
    """
    Orchestra策略的简化版PYU包装类
    用于快速测试和演示
    """
    
    def __init__(self, builder_base, random_seed=None, skip_bn=False, **kwargs):
        # 简化版使用更少的聚类数量和更简单的参数
        kwargs.setdefault('num_local_clusters', 10)
        kwargs.setdefault('num_global_clusters', 5)
        kwargs.setdefault('memory_size', 256)
        kwargs.setdefault('temperature', 0.5)
        
        logging.warning(f"PYUOrchestraSimpleStrategy.__init__ called. builder_base type: {type(builder_base)}, kwargs keys: {list(kwargs.keys())}")
        super().__init__(builder_base=builder_base, random_seed=random_seed, skip_bn=skip_bn, **kwargs)

