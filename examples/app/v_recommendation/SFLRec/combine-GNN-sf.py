import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import to_undirected
import copy
from scipy.sparse import load_npz
# 1. 数据预处理

class MovieLensDataProcessor:
    def process(self):
        # 加载预训练嵌入
        user_emb = np.load('/root/project/sf_HAN_torch/douban-HAN/recommend/user_embedding.npy')
        movie_emb = np.load('/root/project/sf_HAN_torch/douban-HAN/recommend/movie_embedding.npy')
        user_movie_matrix = load_npz("/root/project/sf_HAN_torch/douban-HAN/recommend/user_movie_matrix.npz")

        num_users = user_emb.shape[0]  # 用户向量总数
        num_movies = movie_emb.shape[0]  # 电影向量总数
  
        # 构建特征矩阵
        x = torch.FloatTensor(np.vstack([user_emb, movie_emb]))
        user_indices, movie_indices = user_movie_matrix.nonzero()
        movie_indices += num_users  # 偏移电影索引

        # 合并为单一numpy数组
        np_edge = np.array([user_indices, movie_indices])  # 显式创建二维数组

        # 转换为tensor（更高效的方式）
        edge_index = torch.as_tensor(np_edge, dtype=torch.long)  # 使用as_tensor避免拷贝

        ratings = user_movie_matrix.data  # 直接获取评分值
    
        # 构建边属性
        edge_attr = torch.FloatTensor(ratings)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_users=num_users,
            num_movies=num_movies
        )


# 2.1. 客户端模型
class ClientGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, privacy_budget=1.0):
        super().__init__()
        # 前向部分
        self.front_model = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 后向部分（包括损失计算）
        self.back_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 添加预测层
        self.pred_layer = nn.Linear(hidden_dim * 2, 1)  # 修改这里，用于组合用户和电影特征
        self.privacy_budget = privacy_budget
        
    def forward_front(self, x, edge_index):
        """前向传播第一部分"""
        h = self.front_model[0](x, edge_index)
        h = self.front_model[1](h)
        # 添加差分隐私噪声
        smashed_data = self.add_noise(h)
        return smashed_data
    
    def forward_back(self, server_output, edge_index):
        """前向传播后半部分"""
        h = self.back_model(server_output)
        return h
    
    def predict(self, node_embeddings, edge_index):
        """预测评分"""
        # 获取源节点（用户）和目标节点（电影）的嵌入
        src, dst = edge_index
        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]
        
        # 连接用户和电影的嵌入
        combined = torch.cat([src_emb, dst_emb], dim=1)
        
        # 预测评分
        pred = self.pred_layer(combined)
        return pred.squeeze()
    
    def add_noise(self, x):
        """添加拉普拉斯噪声"""
        sensitivity = 1.0
        noise_scale = sensitivity / self.privacy_budget
        noise = torch.tensor(np.random.laplace(0, noise_scale, x.shape))
        return x + noise.float()

# 2.2 带知识蒸馏的客户端类
class DistillationClient:
    def __init__(self, input_dim, hidden_dim, privacy_budget=1.0):
        self.private_model = ClientGNN(input_dim, hidden_dim, privacy_budget)
        self.shared_model = ClientGNN(input_dim, hidden_dim, privacy_budget)
        self.temperature = 3.0
        
    def forward_private(self, x, edge_index):
        return self.private_model(x, edge_index)
    
    def forward_shared(self, x, edge_index):
        return self.shared_model(x, edge_index)
    
    def compute_distillation_loss(self, x, edge_index):
        with torch.no_grad():
            private_output = self.private_model(x, edge_index)
        shared_output = self.shared_model(x, edge_index)
        
        soft_private = F.softmax(private_output / self.temperature, dim=1)
        soft_shared = F.softmax(shared_output / self.temperature, dim=1)
        
        distill_loss = F.kl_div(
            F.log_softmax(shared_output / self.temperature, dim=1),
            soft_private,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distill_loss
    


# 3 异构数据
class DataSplitter:
    def __init__(self, data):
        self.data = data
        
    def split_data(self):
        # 获取所有评分数据的索引
        num_interactions = self.data.edge_index.shape[1]
        indices = torch.randperm(num_interactions)
        
        # 60% 给Alice, 40% 给Bob
        alice_size = int(0.6 * num_interactions)
        
        # 分割边索引和边属性
        alice_indices = indices[:alice_size]
        bob_indices = indices[alice_size:]
        
        # 创建Alice的数据
        alice_data = Data(
            x=self.data.x,
            edge_index=self.data.edge_index[:, alice_indices],
            edge_attr=self.data.edge_attr[alice_indices],
            num_users=self.data.num_users,
            num_movies=self.data.num_movies
        )
        
        # 创建Bob的数据
        bob_data = Data(
            x=self.data.x,
            edge_index=self.data.edge_index[:, bob_indices],
            edge_attr=self.data.edge_attr[bob_indices],
            num_users=self.data.num_users,
            num_movies=self.data.num_movies
        )
        
        return alice_data, bob_data
    def split_client_data(self, client_data, train_ratio=0.7, val_ratio=0.15):
        """为每个客户端的数据进行训练集/验证集/测试集拆分"""
        # 获取该客户端的所有唯一用户
        unique_users = torch.unique(client_data.edge_index[0])
        num_users = len(unique_users)
        
        # 随机打乱用户顺序
        shuffled_users = unique_users[torch.randperm(num_users)]
        
        # 计算每个集合的用户数量
        train_size = int(num_users * train_ratio)
        val_size = int(num_users * val_ratio)
        
        # 划分用户
        train_users = shuffled_users[:train_size]
        val_users = shuffled_users[train_size:train_size + val_size]
        test_users = shuffled_users[train_size + val_size:]
        
        # 创建掩码
        train_mask = torch.isin(client_data.edge_index[0], train_users)
        val_mask = torch.isin(client_data.edge_index[0], val_users)
        test_mask = torch.isin(client_data.edge_index[0], test_users)
        
        # 创建数据集
        train_data = Data(
            x=client_data.x,
            edge_index=client_data.edge_index[:, train_mask],
            edge_attr=client_data.edge_attr[train_mask],
            num_users=client_data.num_users,
            num_movies=client_data.num_movies
        )
        
        val_data = Data(
            x=client_data.x,
            edge_index=client_data.edge_index[:, val_mask],
            edge_attr=client_data.edge_attr[val_mask],
            num_users=client_data.num_users,
            num_movies=client_data.num_movies
        )
        
        test_data = Data(
            x=client_data.x,
            edge_index=client_data.edge_index[:, test_mask],
            edge_attr=client_data.edge_attr[test_mask],
            num_users=client_data.num_users,
            num_movies=client_data.num_movies
        )
        
        return train_data, val_data, test_data

# 4.1 Main Server模型
class MainServer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.middle_layers = SAGEConv(hidden_dim, hidden_dim)
        
    def forward(self, smashed_data, edge_index):
        """处理中间层计算"""
        return self.middle_layers(smashed_data, edge_index)
# 4.2 加知识蒸馏的main server
class DistillationServer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.main_server = MainServer(hidden_dim)
        self.temperature = 2.0
        # 添加投影层，用于将输出映射到统一的特征空间
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    def forward(self, smashed_data, edge_index):
        return self.main_server(smashed_data, edge_index)
    
    def compute_consistency_loss(self, alice_output, bob_output):
        soft_alice = F.softmax(alice_output / self.temperature, dim=1)
        soft_bob = F.softmax(bob_output / self.temperature, dim=1)
        
        loss = F.kl_div(
            F.log_softmax(alice_output / self.temperature, dim=1),
            soft_bob,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss
    def project(self, x):
        # 投影到统一特征空间
        return self.projection(x)
# 5. Fed Server（全局服务器）
class FedServer:
    def __init__(self, input_dim, hidden_dim):
        self.global_shared_model = ClientGNN(input_dim, hidden_dim)
        self.main_server = MainServer(hidden_dim)
        
    def aggregate_shared_models(self, clients):
        """执行联邦平均 - 只聚合shared models"""
        global_state = self.global_shared_model.state_dict()
        
        # 收集所有客户端的共享模型参数
        for key in global_state.keys():
            temp = torch.stack([
                client.shared_model.state_dict()[key]
                for client in clients
            ])
            global_state[key] = torch.mean(temp, dim=0)
            
        # 更新全局模型
        self.global_shared_model.load_state_dict(global_state)
        return copy.deepcopy(self.global_shared_model.state_dict())

# 6. Distillation Coordinator
class DistillationCoordinator:
    def __init__(self, input_dim, hidden_dim):
        # 客户端
        self.alice = DistillationClient(input_dim, hidden_dim)
        self.bob = DistillationClient(input_dim, hidden_dim)
        
        # 服务器组件
        self.main_server = DistillationServer(hidden_dim)
        self.fed_server = FedServer(input_dim, hidden_dim)
        
        # 优化器
        self.alice_private_optimizer = torch.optim.Adam(self.alice.private_model.parameters())
        self.alice_shared_optimizer = torch.optim.Adam(self.alice.shared_model.parameters())
        self.bob_private_optimizer = torch.optim.Adam(self.bob.private_model.parameters())
        self.bob_shared_optimizer = torch.optim.Adam(self.bob.shared_model.parameters())
        self.server_optimizer = torch.optim.Adam(self.main_server.parameters())
        
        self.current_round = 0
        
        self.criterion = nn.MSELoss()
        self.distill_alpha = 0.3
    def train_private_models(self, alice_data, bob_data):
        """训练客户端的私有模型"""
        total_private_loss = 0.0
        
        # 训练 Alice 的私有模型
        self.alice_private_optimizer.zero_grad()
        # 前向传播
        alice_front_output = self.alice.private_model.forward_front(
            alice_data.x, 
            alice_data.edge_index
        )
        # 通过服务器处理
        alice_server_output = self.main_server(alice_front_output, alice_data.edge_index)
        # 后向传播
        alice_final_output = self.alice.private_model.forward_back(
            alice_server_output, 
            alice_data.edge_index
        )
        # 预测和损失计算
        alice_pred = self.alice.private_model.predict(
            alice_final_output,
            alice_data.edge_index
        )
        alice_loss = self.criterion(alice_pred, alice_data.edge_attr)
        alice_loss.backward()
        self.alice_private_optimizer.step()
        total_private_loss += alice_loss.item()

        # 训练 Bob 的私有模型
        self.bob_private_optimizer.zero_grad()
        # 前向传播
        bob_front_output = self.bob.private_model.forward_front(
            bob_data.x, 
            bob_data.edge_index
        )
        # 通过服务器处理
        bob_server_output = self.main_server(bob_front_output, bob_data.edge_index)
        # 后向传播
        bob_final_output = self.bob.private_model.forward_back(
            bob_server_output, 
            bob_data.edge_index
        )
        # 预测和损失计算
        bob_pred = self.bob.private_model.predict(
            bob_final_output,
            bob_data.edge_index
        )
        bob_loss = self.criterion(bob_pred, bob_data.edge_attr)
        bob_loss.backward()
        self.bob_private_optimizer.step()
        total_private_loss += bob_loss.item()

        return total_private_loss / 2.0

    def train_shared_models(self, alice_data, bob_data):
        """训练客户端的共享模型（包括知识蒸馏）"""
        total_shared_loss = 0.0
        
        # 训练 Alice 的共享模型
        self.alice_shared_optimizer.zero_grad()
        # 获取私有模型的输出（用于知识蒸馏）
        with torch.no_grad():
            alice_private_front = self.alice.private_model.forward_front(
                alice_data.x, 
                alice_data.edge_index
            )
            alice_private_server = self.main_server(
                alice_private_front, 
                alice_data.edge_index
            )
            alice_private_output = self.alice.private_model.forward_back(
                alice_private_server, 
                alice_data.edge_index
            )
        
        # 共享模型的前向传播
        alice_shared_front = self.alice.shared_model.forward_front(
            alice_data.x, 
            alice_data.edge_index
        )
        alice_shared_server = self.main_server(
            alice_shared_front, 
            alice_data.edge_index
        )
        alice_shared_output = self.alice.shared_model.forward_back(
            alice_shared_server, 
            alice_data.edge_index
        )
        
        # 计算预测损失和蒸馏损失
        alice_pred = self.alice.shared_model.predict(
            alice_shared_output,
            alice_data.edge_index
        )
        alice_pred_loss = self.criterion(alice_pred, alice_data.edge_attr)
        alice_distill_loss = F.mse_loss(
            alice_shared_output,
            alice_private_output.detach()
        )
        alice_total_loss = alice_pred_loss + self.distill_alpha * alice_distill_loss
        alice_total_loss.backward()
        self.alice_shared_optimizer.step()
        total_shared_loss += alice_total_loss.item()

        # 训练 Bob 的共享模型
        self.bob_shared_optimizer.zero_grad()
        # 获取私有模型的输出（用于知识蒸馏）
        with torch.no_grad():
            bob_private_front = self.bob.private_model.forward_front(
                bob_data.x, 
                bob_data.edge_index
            )
            bob_private_server = self.main_server(
                bob_private_front, 
                bob_data.edge_index
            )
            bob_private_output = self.bob.private_model.forward_back(
                bob_private_server, 
                bob_data.edge_index
            )
        
        # 共享模型的前向传播
        bob_shared_front = self.bob.shared_model.forward_front(
            bob_data.x, 
            bob_data.edge_index
        )
        bob_shared_server = self.main_server(
            bob_shared_front, 
            bob_data.edge_index
        )
        bob_shared_output = self.bob.shared_model.forward_back(
            bob_shared_server, 
            bob_data.edge_index
        )
        
        # 计算预测损失和蒸馏损失
        bob_pred = self.bob.shared_model.predict(
            bob_shared_output,
            bob_data.edge_index
        )
        bob_pred_loss = self.criterion(bob_pred, bob_data.edge_attr)
        bob_distill_loss = F.mse_loss(
            bob_shared_output,
            bob_private_output.detach()
        )
        bob_total_loss = bob_pred_loss + self.distill_alpha * bob_distill_loss
        bob_total_loss.backward()
        self.bob_shared_optimizer.step()
        total_shared_loss += bob_total_loss.item()

        return total_shared_loss / 2.0

    def train_server(self, alice_data, bob_data):
        """训练服务器端一致性"""
        self.server_optimizer.zero_grad()
        
        # 获取 Alice 和 Bob 的共享模型输出
        alice_front = self.alice.shared_model.forward_front(
            alice_data.x, 
            alice_data.edge_index
        )
        bob_front = self.bob.shared_model.forward_front(
            bob_data.x, 
            bob_data.edge_index
        )
        
        # 服务器处理
        alice_server = self.main_server(alice_front, alice_data.edge_index)
        bob_server = self.main_server(bob_front, bob_data.edge_index)
        
        # 计算一致性损失
        # 这里我们使用 MSE 损失来衡量两个客户端输出的一致性
        # 为了计算损失，我们需要将输出投影到相同的维度
        alice_proj = self.main_server.project(alice_server)
        bob_proj = self.main_server.project(bob_server)
        
        consistency_loss = F.mse_loss(alice_proj, bob_proj)
        consistency_loss.backward()
        self.server_optimizer.step()
        
        return consistency_loss.item()
    def train_round(self, alice_data, bob_data):
        self.current_round += 1
        
        # 1. 训练私有模型
        private_loss = self.train_private_models(alice_data, bob_data)
        
        # 2. 训练共享模型（包括知识蒸馏）
        shared_loss = self.train_shared_models(alice_data, bob_data)
        
        # 3. 服务器端一致性训练
        server_loss = self.train_server(alice_data, bob_data)
        
        return private_loss, shared_loss, server_loss
        
    def evaluate(self, data, model_type='shared'):
        """评估模型性能
        
        Args:
            data: 要评估的数据
            model_type: 'shared' 或 'private'，指定评估哪种模型
        
        Returns:

            float: MSE损失值
        """
        alice_data, bob_data = data

        self.alice.private_model.eval()
        self.alice.shared_model.eval()
        self.bob.private_model.eval()
        self.bob.shared_model.eval()
        self.main_server.eval()
        
        with torch.no_grad():
            # 选择要评估的模型
            if model_type == 'shared':
                alice_model = self.alice.shared_model
                bob_model = self.bob.shared_model
            else:
                alice_model = self.alice.private_model
                bob_model = self.bob.private_model
            
            # 前向传播
            alice_front = alice_model.forward_front(alice_data.x, alice_data.edge_index)
            bob_front = bob_model.forward_front(bob_data.x, bob_data.edge_index)
            server_output_alice = self.main_server(alice_front, alice_data.edge_index)
            server_output_bob = self.main_server(bob_front, bob_data.edge_index)
            alice_output = alice_model.forward_back(server_output_alice, alice_data.edge_index)
            bob_output = bob_model.forward_back(server_output_bob, bob_data.edge_index)
            predictions_alice = alice_model.predict(alice_output, alice_data.edge_index)
            predictions_bob = bob_model.predict(bob_output, bob_data.edge_index)
            # 计算MSE mae损失
            alice_mse_loss = F.mse_loss(predictions_alice, alice_data.edge_attr) 
            bob_mse_loss = F.mse_loss(predictions_bob, bob_data.edge_attr)
            alice_mae = F.l1_loss(predictions_alice, alice_data.edge_attr)
            bob_mae = F.l1_loss(predictions_bob, bob_data.edge_attr)
                        
        self.alice.private_model.train()
        self.alice.shared_model.train()
        self.bob.private_model.train()
        self.bob.shared_model.train()
        self.main_server.train()
        
        return {
            'alice_mse': alice_mse_loss.item(),
            'bob_mse': bob_mse_loss.item(),
            'alice_mae': alice_mae.item(),
            "bob_mae" : bob_mae.item()
        }
    def fed_sync(self, sync_frequency=5):
        """执行联邦同步"""
        if self.current_round % sync_frequency == 0:
            # 执行联邦平均
            global_state = self.fed_server.aggregate_shared_models([self.alice, self.bob])
            
            # 更新所有客户端的共享模型
            self.alice.shared_model.load_state_dict(global_state)
            self.bob.shared_model.load_state_dict(global_state)
            
            print(f"Round {self.current_round}: Performed federated synchronization")

def train_distillation_system(coordinator, train_data, val_data, test_data, num_epochs=100, sync_frequency=5, patience=50):
    """
    训练系统，包含验证和提前停止机制
    
    Args:
        coordinator: DistillationCoordinator实例
        train_data: (alice_train, bob_train) 元组
        val_data: (alice_val, bob_val) 元组
        test_data: (alice_test, bob_test) 元组
        num_epochs: 训练轮数
        sync_frequency: 联邦同步频率
        patience: 提前停止的耐心值
    """
    alice_train, bob_train = train_data
    alice_val, bob_val = val_data
    alice_test, bob_test = test_data
    
    best_val_mse = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_mse': [],
        'private_loss': [],
        'shared_loss': [],
        'server_loss': []
    }
    for epoch in range(num_epochs):
        # 训练阶段
        private_loss, shared_loss, server_loss = coordinator.train_round(alice_train, bob_train)
        
        # 执行联邦同步
        coordinator.fed_sync(sync_frequency)
        
        # 验证阶段
        val_metrics = coordinator.evaluate((alice_val, bob_val), model_type='private')
        val_mse = (val_metrics['bob_mse']+ val_metrics["alice_mse"]) / 2
        
        # 记录历史
        history['train_loss'].append((private_loss + shared_loss) / 2)
        history['val_mse'].append(val_mse)
        history['private_loss'].append(private_loss)
        history['shared_loss'].append(shared_loss)
        history['server_loss'].append(server_loss)
        
        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, \n'
                  f'Train Loss: {(private_loss + shared_loss) / 2:.4f}, \n'
                  f'Val MSE: \nAlice \t{val_metrics["alice_mse"]:.4f}, Bob \t{val_metrics["bob_mse"]:.4f}\n'
                  f'Val MAE: \nAlice \t{val_metrics["alice_mae"]:.4f}, Bob \t{val_metrics["bob_mae"]:.4f}\n')
        
        # 检查是否需要保存最佳模型
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = {
                'alice_shared': copy.deepcopy(coordinator.alice.shared_model.state_dict()),
                'bob_shared': copy.deepcopy(coordinator.bob.shared_model.state_dict()),
                'main_server': copy.deepcopy(coordinator.main_server.state_dict()),
                'epoch': epoch
            }
        else:
            patience_counter += 1
            
        # 提前停止检查
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # 加载最佳模型状态
    if best_model_state is not None:
        coordinator.alice.shared_model.load_state_dict(best_model_state['alice_shared'])
        coordinator.bob.shared_model.load_state_dict(best_model_state['bob_shared'])
        coordinator.main_server.load_state_dict(best_model_state['main_server'])
        print(f"Loaded best model from epoch {best_model_state['epoch']+1}")
    
    # 在测试集上进行最终评估
    test_metrics = coordinator.evaluate((alice_test, bob_test), model_type='shared')
    print("\nTest Set Results:\n")
    print(f"MSE:\n Alice \t {test_metrics['alice_mse']:.4f}, Bob \t{test_metrics['bob_mse']:.4f}\n")
    print(f"MAE:\n Alice \t{test_metrics['alice_mae']:.4f}, Bob \t{test_metrics['bob_mae']:.4f}\n")
    
    return history, test_metrics


if __name__ == '__main__':
    # 1. 数据处理和加载

    # data_processor = MovieLensDataProcessor(
    #     '/root/project/smashed_data_gnn/ml-1m/ratings.dat',
    #     '/root/project/smashed_data_gnn/ml-1m/movies.dat',
    #     '/root/project/smashed_data_gnn/ml-1m/users.dat'
    # )
    data_processor = MovieLensDataProcessor()
    data = data_processor.process()

    # 2. 数据分割
    splitter = DataSplitter(data)
    
    # 3. 首先将数据分给Alice和Bob
    alice_data, bob_data = splitter.split_data()

    # 4. 对Alice和Bob的数据进行训练/验证/测试集拆分
    alice_train, alice_val, alice_test = splitter.split_client_data(alice_data)
    bob_train, bob_val, bob_test = splitter.split_client_data(bob_data)
    
    # 5. 创建协调器
    feature_dim = data.x.shape[1]
    hidden_dim = 64
    coordinator = DistillationCoordinator(feature_dim, hidden_dim)
    
    # 6. 训练系统
    history, test_metrics = train_distillation_system(
        coordinator=coordinator,
        train_data=(alice_train, bob_train),
        val_data=(alice_val, bob_val),
        test_data=(alice_test, bob_test),
        num_epochs=100,
        sync_frequency=5,
        patience=50
    )
    
    # # 7. 可选：绘制训练历史
    # import matplotlib.pyplot as plt
    
    # plt.figure(figsize=(12, 4))
    
    # # 训练和验证损失
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['val_mse'], label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    
    # # 各种损失的变化
    # plt.subplot(1, 2, 2)
    # plt.plot(history['private_loss'], label='Private Loss')
    # plt.plot(history['shared_loss'], label='Shared Loss')
    # plt.plot(history['server_loss'], label='Server Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Component Losses')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.savefig("./train_log.png")