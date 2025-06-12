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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DualDomainSeqDataset, collate_fn_enhance
from model import BERT4Rec, BAHE  # 导入 BAHE 模块
from sklearn.metrics import roc_auc_score
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path


# 初始化日志
def init_logger(log_dir, log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(Path(log_dir) / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def test(model, bahe_model, val_loader):
    model.eval()
    bahe_model.eval()
    total_loss = 0
    preds, labels = [], []

    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing"):
            # 获取数据
            user_node = batch['user_node'].long()
            item_node = batch['i_node'].long()
            seq_d1 = batch['seq_d1'].long()
            seq_d2 = batch['seq_d2'].long()
            domain_id = batch['domain_id'].long()
            label = batch['label'].float()  # [batch_size]
            behavior_texts = batch['behavior_texts']

            # 获取用户嵌入
            user_embedding = bahe_model(behavior_texts)
            # 按域选择正确输出参与损失

            output = model(user_embedding, item_node, seq_d1, seq_d2, domain_id)
            output = output.view(-1)  # 进行必要的形状调整
            loss = criterion(output, label)
            total_loss += loss.item()

            # 收集预测结果用于 AUC
            preds.extend(output.cpu().numpy())
            labels.extend(label.cpu().numpy())

    # 计算平均损失和 AUC
    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(labels, preds)

    return avg_loss, auc


# 训练函数
def train(model, bahe_model, train_loader, val_loader, optimizer, args):
    logger = init_logger(args.model_dir, args.log_file)
    best_auc = 0

    for epoch in range(args.epochs):
        model.train()
        bahe_model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            # 获取数据
            # 获取数据并确保类型正确
            user_node = batch['user_node'].long()
            item_node = batch['i_node'].long()
            seq_d1 = batch['seq_d1'].long()
            seq_d2 = batch['seq_d2'].long()
            domain_id = batch['domain_id'].long()
            label = batch['label'].float()
            behavior_texts = batch['behavior_texts']
            # 打印维度信息（调试用）
            print(f"Sequence shapes - seq_d1: {seq_d1.shape}, seq_d2: {seq_d2.shape}")
            print(f"Item node shape: {item_node.shape}")
            print(f"Domain ID shape: {domain_id.shape}")
            # 使用较小的批次处理行为文本
            user_embedding = bahe_model(behavior_texts)

            # 模型预测
            optimizer.zero_grad()
            # output_d1, output_d2 = model(user_embedding, item_node, seq_d1, seq_d2, domain_id)
            output = model(user_embedding, item_node, seq_d1, seq_d2, domain_id)

            output = output.view(-1)  # 进行必要的形状调整
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(output, label)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 定期清理内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()  # 即使在CPU上运行也是安全的

        # 每个epoch结束后验证
        avg_loss = total_loss / len(train_loader)
        val_loss, val_auc = test(model, bahe_model, val_loader)

        # 记录日志
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    'bert4rec_state_dict': model.state_dict(),
                    'bahe_state_dict': bahe_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_auc': best_auc,
                },
                Path(args.model_dir) / "best_model.pt",
            )

    logger.info(f"Training finished. Best AUC: {best_auc:.4f}")


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT4Rec Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument(
        "--model_dir", type=str, default="model", help="Directory to save models"
    )
    parser.add_argument(
        "--log_file", type=str, default="train.log", help="Log file name"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="amazon_dataset", help="Path to dataset"
    )
    parser.add_argument('-ds', '--dataset_type', type=str, default='amazon')
    parser.add_argument('-dm', '--domain_type', type=str, default='cloth_sport')

    parser.add_argument(
        '--long_length',
        type=int,
        default=7,
        help='the length for setting long-tail node',
    )
    parser.add_argument(
        '--neg_nums', type=int, default=199, help='sample negative numbers'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=0.25,
        help='overlap ratio for choose dataset ',
    )
    # overlap_ratio = 0.25 表示25%的用户是跨域用户（即在两个域都有行为的用户）
    parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--bs', type=int, default=256, help='# images in batch')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden layer dim')
    parser.add_argument('--isInC', type=bool, default=False, help='add inc ')
    parser.add_argument('--isItC', type=bool, default=False, help='add itc')
    parser.add_argument('--ts1', type=float, default=0.5, help='mask rate for encoder')
    parser.add_argument('--ts2', type=float, default=0.5, help='mask rate for decoder')
    args = parser.parse_args()
    user_length = 895510  # 63275#6814 cdr23 #63275 cdr12
    item_length_d1 = 8240
    item_length_d2 = 26272
    item_length = 447410  # item_length_d1 + item_length_d2 + 1 + 20000#1739+2 #13713 cdr23 #1739 + 2#item_length_d1 + item_length_d2 + 1 + 20000#1739 + 1 +200 # 1 = pad item #item_length_d1 + item_length_d2 + 1 + 20000

    # 设备设置
    device = torch.device('cpu')

    # 加载数据集

    datasetTrain = DualDomainSeqDataset(
        seq_len=args.seq_len,
        isTrain=True,
        neg_nums=args.neg_nums,
        long_length=args.long_length,
        pad_id=item_length - 1,
        csv_path=f"{args.dataset_type}_dataset/{args.domain_type}_train{int(args.overlap_ratio*100)}.csv",
    )
    trainLoader = DataLoader(
        datasetTrain,
        batch_size=32,  # 使用更小的批次大小
        shuffle=True,
        num_workers=4,  # 减少工作进程数
        drop_last=True,
        collate_fn=collate_fn_enhance,
    )

    datasetVal = DualDomainSeqDataset(
        seq_len=args.seq_len,
        isTrain=False,
        neg_nums=args.neg_nums,
        long_length=args.long_length,
        pad_id=item_length - 1,
        csv_path=f"{args.dataset_type}_dataset/{args.domain_type}_test.csv",
    )
    valLoader = DataLoader(
        datasetVal,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn_enhance,
    )

    # 初始化模型
    bahe_model = BAHE(
        albert_model_name='albert-base-v2',
        embed_dim=args.emb_dim,
        num_heads=4,
        ff_dim=512,
        num_layers=2,
    )

    model = BERT4Rec(
        user_length=user_length,
        user_emb_dim=args.emb_dim,
        item_length=item_length,
        item_emb_dim=args.emb_dim,
        seq_len=args.seq_len,
        hid_dim=args.hid_dim,
        bs=32,  # 更新批次大小
        isInC=args.isInC,
        isItC=args.isItC,
        threshold1=args.ts1,
        threshold2=args.ts2,
    )

    # 优化器
    optimizer = optim.Adam(
        list(model.parameters()) + list(bahe_model.parameters()), lr=args.lr
    )

    # 训练模型
    train(model, bahe_model, trainLoader, valLoader, optimizer, args)
