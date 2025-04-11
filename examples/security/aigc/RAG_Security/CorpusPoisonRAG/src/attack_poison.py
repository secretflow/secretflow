import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
import random
from tqdm import tqdm

# 导入datasets库中的Dataset类
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
import torch.nn.functional as F

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer

# 获取日志记录器
logger = logging.getLogger(__name__)

import argparse
# 导入beir库中的util和GenericDataLoader
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from collections import Counter

# 导入utils模块中的load_models函数
from utils import load_models

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    这个类用于存储给定PyTorch模块输出的中间梯度，这些梯度在默认情况下可能不会被保留。
    """
    def __init__(self, module):
        # 梯度
        self._stored_gradient = None
        # 注册一个钩子函数来存储梯度
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        # 存储梯度
        self._stored_gradient = grad_out[0]

    def get(self):
        # 返回存储的梯度
        return self._stored_gradient

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    """返回词片嵌入模块。"""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    # 根据不同的模型类型获取嵌入层
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    """返回候选替换词的顶级选择。"""
    # 计算梯度与嵌入矩阵的点积
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

    # f(a) --> f(b)  =  f'(a) * (b - a) = f'(a) * b

def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator, device='cuda'):
    """Returns the 2-way classification accuracy (used during training)"""
    """返回二元分类准确率（在训练期间使用）"""
    # 设置模型为评估模式
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    # 遍历数据加载器中的所有数据
    for idx, (data) in tqdm(enumerate(dataloader)):
        data = data_collator(data) # [bsz, 3, max_len]

        # Get query embeddings
        # 获取查询嵌入
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent)  # [b x d]

        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass) # [b x d]

        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()

        p_sent = {'input_ids': adv_passage_ids, 
                  'attention_mask': adv_passage_attention, 
                  'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)  # [k x d]

        sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]

        acc += (sim_to_gold > sim).sum().cpu().item()
        tot += q_emb.shape[0]
    
    print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
    return acc / tot

def kmeans_split(data_dict, model, get_emb, tokenizer, k, split):
    """Get all query embeddings and perform kmeans"""
    """获取所有查询嵌入并执行kmeans"""
    # get query embs
    # 获取查询嵌入
    q_embs = []
    for q in tqdm(data_dict["sent0"]):
        query_input = tokenizer(q, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
        q_embs.append(query_emb[0].cpu().numpy())
    q_embs = np.array(q_embs)
    # print("q_embs", q_embs.shape)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    # print(Counter(kmeans.labels_))

    ret_dict = {"sent0": [], "sent1": []}
    for i in range(len(data_dict["sent0"])):
        if kmeans.labels_[i] == split:
            ret_dict["sent0"].append(data_dict["sent0"][i])
            ret_dict["sent1"].append(data_dict["sent1"][i])
    # print("K = %d, split = %d, tot num = %d"%(k, split, len(ret_dict["sent0"])))

    return ret_dict

# 主函数
def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)

    parser.add_argument("--num_adv_passage_tokens", default=50, type=int)
    parser.add_argument("--num_cand", default=100, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--num_iter", default=5000, type=int)
    parser.add_argument("--num_grad_iter", default=1, type=int)

    parser.add_argument("--output_file", default=None, type=str)

    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--kmeans_split", default=0, type=int)
    parser.add_argument("--do_kmeans", default=False, action="store_true")

    parser.add_argument("--dont_init_gold", action="store_true", help="if ture, do not init with gold passages")
    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(0)
    
    # Load models
    model, c_model, tokenizer, get_emb = load_models(args.model_code)
        
        
    # 设置模型为评估模式并移动到指定设备
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)

    # Load datasets
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)
    l = list(qrels.items())
    random.shuffle(l)
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)
    
    # do kmeans
    # 如果需要，执行kmeans聚类
    if args.do_kmeans:
        data_dict = kmeans_split(data_dict, model, get_emb, tokenizer, k=args.k, split=args.kmeans_split)
    
    datasets = {"train": Dataset.from_dict(data_dict)}

    # 定义tokenization函数
    def tokenization(examples):
        q_feat = tokenizer(examples["sent0"], max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)
        c_feat = tokenizer(examples["sent1"], max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)

        ret = {}
        for key in q_feat:
            ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]

        return ret

    # use 30% examples as dev set during training
    # 使用30%的样本作为开发集
    print('Train data size = %d'%(len(datasets["train"])))
    num_valid = min(1000, int(len(datasets["train"]) * 0.3))
    datasets["subset_valid"] = Dataset.from_dict(datasets["train"][:num_valid])
    datasets["subset_train"] = Dataset.from_dict(datasets["train"][num_valid:])

    train_dataset = datasets["subset_train"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    dataset = datasets["subset_valid"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    print('Finished loading datasets')

    data_collator = default_data_collator
    dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=True, collate_fn=lambda x: x )
    valid_dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x )

    # Set up variables for embedding gradients
    # 设置嵌入梯度变量
    embeddings = get_embeddings(c_model)
    print('Model embedding', embeddings)
    embedding_gradient = GradientStorage(embeddings)

    # Initialize adversarial passage
    # 初始化对抗性段落
    adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens
    print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
    adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
    adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

    best_adv_passage_ids = adv_passage_ids.clone()
    best_acc = evaluate_acc(model, c_model, get_emb, valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
    print(best_acc)

    for it_ in range(args.num_iter):
        print(f"Iteration: {it_}")
        
        # print(f'Accumulating Gradient {args.num_grad_iter}')
        c_model.zero_grad()

        pbar = range(args.num_grad_iter)
        train_iter = iter(dataloader)
        grad = None

        for _ in pbar:
            try:
                data = next(train_iter)
                data = data_collator(data) # [bsz, 3, max_len]
            except:
                print('Insufficient data!')
                break
        
            q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
            q_emb = get_emb(model, q_sent).detach()

            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
            gold_emb = get_emb(c_model, gold_pass).detach()

            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
            sim_to_gold_mean = sim_to_gold.mean().cpu().item()
            # print('Avg sim to gold p =', sim_to_gold_mean)

            # Initialize the adversarial passage with a gold passage
            # 如果是第一次迭代，并且当前最佳准确率为1.0，并且没有指定不使用金标准段落初始化，则使用金标准段落初始化对抗性段落
            if it_ == 0 and _ == 0 and best_acc == 1.0 and (not args.dont_init_gold):
                print("Init with a gold passage")
                ll = min(len(gold_pass['input_ids'][0]), args.num_adv_passage_tokens)
                adv_passage_ids[0][:ll] = gold_pass['input_ids'][0][:ll]
                print(adv_passage_ids.shape)
                print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))

                best_adv_passage_ids = adv_passage_ids.clone()
                best_acc = evaluate_acc(model, c_model, get_emb, valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
                print(best_acc)

            p_sent = {'input_ids': adv_passage_ids, 
                    'attention_mask': adv_passage_attention, 
                    'token_type_ids': adv_passage_token_type}
            p_emb = get_emb(c_model, p_sent)

            # Compute loss
            sim = torch.mm(q_emb, p_emb.T)  # [b x k]
            # print(it_, _, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)
            suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
            # print('Attack on train: %d / %d'%(suc_att, sim_to_gold.shape[0]), 'best_acc', best_acc)
            loss = sim.mean()
            # print('loss', loss.cpu().item())
            loss.backward()

            temp_grad = embedding_gradient.get()
            if grad is None:
                grad = temp_grad.sum(dim=0) / args.num_grad_iter
            else:
                grad += temp_grad.sum(dim=0) / args.num_grad_iter
            
        # print('Evaluating Candidates')
        pbar = range(args.num_grad_iter)
        train_iter = iter(dataloader)

        token_to_flip = random.randrange(args.num_adv_passage_tokens)
        candidates = hotflip_attack(grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=True,
                                    num_candidates=args.num_cand,
                                    filter=None)
        
        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        current_acc_rate = 0
        candidate_acc_rates = torch.zeros(args.num_cand, device=device)

        for step in pbar:
            try:
                data = next(train_iter)
                data = data_collator(data) # [bsz, 3, max_len]
            except:
                print('Insufficient data!')
                break
                
            q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
            q_emb = get_emb(model, q_sent).detach()

            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
            gold_emb = get_emb(c_model, gold_pass).detach()

            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
            sim_to_gold_mean = sim_to_gold.mean().cpu().item()
            # print('Avg sim to gold p =', sim_to_gold_mean)

            p_sent = {'input_ids': adv_passage_ids, 
                    'attention_mask': adv_passage_attention, 
                    'token_type_ids': adv_passage_token_type}
            p_emb = get_emb(c_model, p_sent)

            # Compute loss
            sim = torch.mm(q_emb, p_emb.T)  # [b x k]
            # print(it_, _, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)
            suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
            # print('Attack on train: %d / %d'%(suc_att, sim_to_gold.shape[0]), 'best_acc', best_acc)
            loss = sim.mean()
            temp_score = loss.sum().cpu().item()

            current_score += temp_score
            current_acc_rate += suc_att

            for i, candidate in enumerate(candidates):
                temp_adv_passage = adv_passage_ids.clone()
                temp_adv_passage[:, token_to_flip] = candidate
                p_sent = {'input_ids': temp_adv_passage, 
                    'attention_mask': adv_passage_attention, 
                    'token_type_ids': adv_passage_token_type}
                p_emb = get_emb(c_model, p_sent)
                with torch.no_grad():
                    sim = torch.mm(q_emb, p_emb.T)
                    can_suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
                    can_loss = sim.mean()
                    temp_score = can_loss.sum().cpu().item()

                    candidate_scores[i] += temp_score
                    candidate_acc_rates[i] += can_suc_att
        # print(current_score, max(candidate_scores).cpu().item())
        # print(current_acc_rate, max(candidate_acc_rates).cpu().item())

        # if find a better one, update
        if (candidate_scores > current_score).any() or (candidate_acc_rates > current_acc_rate).any():
            logger.info('Better adv_passage detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
            # print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
        else:
            print('No improvement detected!')
            continue

        cur_acc = evaluate_acc(model, c_model, get_emb, valid_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
        if cur_acc < best_acc:
            best_acc = cur_acc
            best_adv_passage_ids = adv_passage_ids.clone()
            logger.info('!!! Updated best adv_passage')
            print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))

            if args.output_file is not None:
                with open(args.output_file, 'w') as f:
                    json.dump({"it": it_, "best_acc": best_acc, "dummy": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]), "tot": num_valid}, f)
        
        print('best_acc', best_acc)



if __name__ == "__main__":
    main()

