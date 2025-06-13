# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
from pathlib import Path

import hydra

# import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from utils.metrics import *
from utils.common import *

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = "f01277e9a4a605e59d5c667b58ac365388f90fa0"
# os.environ["WANDB_MODE"] = "online"  # 设置wandb运行模式为离线
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用GPU 1，将其当作GPU 0来使用
os.environ["WANDB_DISABLE"] = "true"


def train(
    model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping
):
    model.train()  # 设置模型为训练模式
    torch.set_grad_enabled(True)  # 开启梯度计算

    sum_loss = torch.zeros(1).to(local_rank)  # 存储累积损失
    sum_auc = torch.zeros(1).to(local_rank)  # 存储累积AUC（Area Under Curve）

    # 遍历数据集进行训练
    for cnt, (
        subgraph,
        mapping_idx,
        candidate_news,
        candidate_entity,
        entity_mask,
        labels,
    ) in enumerate(
        tqdm(
            dataloader,
            total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)),
            desc=f"[{local_rank}] Training",
        ),
        start=1,
    ):

        # 将数据移到指定的GPU上
        subgraph = subgraph.to(local_rank, non_blocking=True)
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
        entity_mask = entity_mask.to(local_rank, non_blocking=True)

        # 使用自动混合精度训练
        with amp.autocast():
            bz_loss, y_hat = model(
                subgraph,
                mapping_idx,
                candidate_news,
                candidate_entity,
                entity_mask,
                labels,
            )

        # 累积梯度
        scaler.scale(bz_loss).backward()

        # 梯度更新周期
        if cnt % cfg.accumulation_steps == 0 or cnt == int(
            cfg.dataset.pos_count / cfg.batch_size
        ):
            # 更新参数
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()  # 更新scaler的值
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()  # 调整学习率
                ## https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
            optimizer.zero_grad(set_to_none=True)  # 清除梯度

        sum_loss += bz_loss.data.float()  # 累加损失
        sum_auc += area_under_curve(labels, y_hat)  # 累加AUC指标

        # ---------------------------------------- 训练日志
        if cnt % cfg.log_steps == 0:
            # if local_rank == 0:
            # wandb.log({"train_loss": sum_loss.item() / cfg.log_steps, "train_auc": sum_auc.item() / cfg.log_steps})
            print(
                '[{}] Ed: {}, average_loss: {:.5f}, average_acc: {:.5f}'.format(
                    local_rank,
                    cnt * cfg.batch_size,
                    sum_loss.item() / cfg.log_steps,
                    sum_auc.item() / cfg.log_steps,
                )
            )
            sum_loss.zero_()  # 清零累积损失
            sum_auc.zero_()  # 清零累积AUC

        # 如果超过一定步数，进行验证
        if (
            cnt
            > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1))
            and cnt % cfg.val_steps == 0
        ):
            res = val(model, local_rank, cfg)
            model.train()  # 验证后切换回训练模式

            if local_rank == 0:
                pretty_print(res)  # 打印验证结果
                # wandb.log(res)  # 记录到wandb

            early_stop, get_better = early_stopping(res['auc'])  # 判断是否提前停止
            if early_stop:
                print("Early Stop.")
                break  # 提前停止
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    # 保存模型并记录最好的AUC
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc']}")
                    # wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                    # "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})


def val(model, local_rank, cfg):
    model.eval()  # 设置模型为评估模式
    dataloader = load_data(
        cfg, mode='val', model=model, local_rank=local_rank
    )  # 加载验证数据
    tasks = []
    with torch.no_grad():  # 评估时不计算梯度
        for cnt, (
            subgraph,
            mappings,
            clicked_entity,
            candidate_input,
            candidate_entity,
            entity_mask,
            labels,
        ) in enumerate(
            tqdm(
                dataloader,
                total=int(cfg.dataset.val_len / cfg.gpu_num),
                desc=f"[{local_rank}] Validating",
            )
        ):
            # 将数据移到GPU上
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(
                local_rank, non_blocking=True
            )
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)

            # 模型评估阶段的处理过程
            scores = model.module.validation_process(
                subgraph,
                mappings,
                clicked_entity,
                candidate_emb,
                candidate_entity,
                entity_mask,
            )

            tasks.append((labels.tolist(), scores))  # 将结果存储起来

    # 使用多进程池计算各项指标
    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)  # 计算每个任务的指标
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T  # 提取各个指标

    # barrier
    torch.distributed.barrier()  # 用于同步不同GPU的进程

    # 汇总不同GPU的结果
    reduced_auc = reduce_mean(
        torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num
    )
    reduced_mrr = reduce_mean(
        torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num
    )
    reduced_ndcg5 = reduce_mean(
        torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num
    )
    reduced_ndcg10 = reduce_mean(
        torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num
    )

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }

    return res


def main_worker(local_rank, cfg):
    # -----------------------------------------环境初始化
    seed_everything(cfg.seed)  # 设置随机种子
    dist.init_process_group(
        backend='nccl',  # NVIDIA提供的高效分布式通信库，为GPU优化
        init_method='tcp://127.0.0.1:23456',  # 所有的进程通过该地址进行通信
        world_size=cfg.gpu_num,  # 总GPU数量
        rank=local_rank,
    )  # 当前进程的GPU编号

    # -----------------------------------------加载数据集和模型
    num_training_steps = int(
        cfg.num_epochs
        * cfg.dataset.pos_count
        / (cfg.batch_size * cfg.accumulation_steps)
    )
    # accumulation_steps内存问题，用小batch——size,但是每个batch之后不更新，而是累积几个step一起更新，从而模拟更大批次的训练
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)  # 计算warmup步数
    train_dataloader = load_data(
        cfg, mode='train', local_rank=local_rank
    )  # 加载训练数据
    model = load_model(cfg).to(local_rank)  # 加载模型并移到对应GPU
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.optimizer.lr
    )  # 初始化优化器

    lr_lambda = lambda step: 1.0 if step > num_warmup_steps else step / num_warmup_steps
    scheduler = LambdaLR(optimizer, lr_lambda)  # 学习率调整

    # ------------------------------------------加载检查点（如果需要）
    if cfg.load_checkpoint:
        file_path = Path(
            f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}.pth"
        )
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )  # 分布式训练
    optimizer.zero_grad(set_to_none=True)  # 清除优化器的梯度
    scaler = amp.GradScaler()  # 混合精度训练

    # ------------------------------------------训练开始
    early_stopping = EarlyStopping(cfg.early_stop_patience)  # 提前停止

    if local_rank == 0:
        # wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
        #           project=cfg.logger.exp_name, name=cfg.logger.run_name)  # 初始化wandb
        print(model)

    # 启动训练
    train(
        model,
        optimizer,
        scaler,
        scheduler,
        train_dataloader,
        local_rank,
        cfg,
        early_stopping,
    )

    # if local_rank == 0:
    # wandb.finish()  # 结束wandb记录


@hydra.main(
    version_base="1.2",
    config_path=os.path.join(get_root(), "configs"),
    config_name="small",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)  # 设置随机种子
    cfg.gpu_num = torch.cuda.device_count()  # 获取可用的GPU数量
    prepare_preprocessed_data(cfg)  # 准备预处理数据
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))  # 启动多进程训练


if __name__ == "__main__":
    main()  # 启动主函数
