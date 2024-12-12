import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader

import os
import json
import sys

import argparse
# 导入信息检索结果评估工具
import pytrec_eval

import torch
import copy

# sys.path.append("./corpus-poisoning/src/contriever")
# sys.path.append("./corpus-poisoning/src/contriever/src")
from contriever import Contriever
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils import load_models

def evaluate_recall(results, qrels, k_values = [1,3,5,10,20,50,100,1000]):
    """评估指定k值的召回率。"""
    # 初始化一个计数字典以跟踪召回率
    cnt = {k: 0 for k in k_values}
    # 遍历每个查询
    for q in results:
        # 获取每个结果的相似度分数
        sims = list(results[q].items())
        # 按分数降序排序
        sims.sort(key=lambda x: x[1], reverse=True)
        # 获取该查询的真实标签
        gt = qrels[q]
        # 标记是否找到任何真实标签
        found = 0
        # 检查前k个结果
        for i, (c, _) in enumerate(sims[:max(k_values)]):
            # 如果候选结果在真实标签中
            if c in gt:
                found = 1
            if (i + 1) in k_values:
                cnt[i + 1] += found
#             print(i, c, found)
    # 计算每个k值的召回率
    recall = {}
    for k in k_values:
        # 计算召回率并四舍五入
        recall[f"Recall@{k}"] = round(cnt[k] / len(results), 5)
    
    return recall

def main():
    parser = argparse.ArgumentParser(description='test')
    # The model and dataset used to generate adversarial passages 
    # 定义命令行参数，用于模型、数据集和评估设置
    parser.add_argument("--attack_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument("--attack_dataset", type=str, default="nq-train", choices=["nq-train", "msmarco", "nq"])
    parser.add_argument("--advp_path", type=str, default="results/advp", help="the path where generated adversarial passages are stored")
    parser.add_argument("--num_advp", type=str, default="50", help="how many adversarial passages are generated (i.e., k in k-means); you may test multiple by passing `--num_advp 1,10,50`")

    # The model and dataset used to evaluate the attack performance (e.g., if eval_model is different from attack_model, it studies attack across different models)
     # 定义评估模型和数据集的参数
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument('--eval_dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')

    # Where to save the evaluation results (attack performance)
    # 保存评估结果的参数
    parser.add_argument("--save_results", action='store_true', default=False)
    parser.add_argument("--eval_res_path", type=str, default="results/attack_results")
    # 输入的最大序列长度
    parser.add_argument('--max_seq_length', type=int, default=128)
    # 填充选项
    parser.add_argument('--pad_to_max_length', default=True)

    args = parser.parse_args()

    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.eval_dataset)


    # 加载数据集
    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        # 根据数据集名称调整分割
        args.split = 'train'
    # 加载语料库、查询和真实标签
    corpus, queries, qrels = data.load(split=args.split)

    # 检查原始BEIR评估结果
    if args.orig_beir_results is None:
        print("Please evaluate on BEIR first -- %s on %s"%(args.eval_model_code, args.eval_dataset))
        
        # Try to get beir eval results from ./beir_results
        # 尝试从预定义路径获取BEIR评估结果
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    
    # 加载原始评估结果
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    
    # 确保真实标签和结果对齐
    assert len(qrels) == len(results)
    # 打印样本总数
    print('Total samples:', len(results))

    # Load models
    # 使用工具函数加载模型
    model, c_model, tokenizer,get_emb = load_models(args.eval_model_code)

    model.eval()
    model.cuda()
    c_model.eval()
    c_model.cuda()

    def evaluate_adv(prefix, k, qrels, results):
        """基于对抗样本评估其对召回率的影响。"""
        # 打印当前评估设置
        print('Prefix = %s, K = %d'%(prefix, k))
         # 用于存储对抗样本的列表
        adv_ps = []
        for s in range(k):
            # 构造文件名
            file_name = "%s/%s-k%d-s%d.json"%(args.advp_path, prefix, k, s)
            # 检查文件是否存在
            if not os.path.exists(file_name):
                print(f"!!!!! {file_name} does not exist!")
                continue          # 如果未找到，跳过
            with open(file_name, 'r') as f:
                # 加载对抗样本
                p = json.load(f)
                # 添加到列表中
                adv_ps.append(p)
        # 打印对抗样本的数量
        print('# adversaria passages', len(adv_ps))
        # 初始化准确率计数
        acc = 0
        # 初始化总计数
        tot = 0
        # 从对抗样本聚合准确率和总数
        for s in range(len(adv_ps)):
            # print(s, adv_ps[s]["it"], adv_ps[s]["best_acc"], adv_ps[s]["tot"])
            # 计算总准确率
            acc += int(adv_ps[s]["tot"] * adv_ps[s]["best_acc"])
            # 增加总计数
            tot += adv_ps[s]["tot"]
        # print("%.3f (%d / %d)"%(acc / tot, acc, tot))
        
        # 创建对结果的深拷贝，以避免修改原始结果
        adv_results = copy.deepcopy(results)
        
        # 将对抗样本中的“dummy”令牌转换为ID
        adv_p_ids = [tokenizer.convert_tokens_to_ids(p["dummy"]) for p in adv_ps]
        # 将对抗样本ID转换为张量并移动到GPU
        adv_p_ids = torch.tensor(adv_p_ids).cuda()
        # 创建与对抗样本ID相同形状的全1张量，用作注意力掩码
        adv_attention = torch.ones_like(adv_p_ids, device='cuda')
        # 创建与对抗样本ID相同形状的全0张量，用作类型ID
        adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
        # 组合输入字典
        adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}

        # 禁用梯度计算以节省内存
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)
        
        adv_qrels = {q: {"adv%d"%(s):1 for s in range(k)} for q in qrels}
        
        for i, query_id in tqdm(enumerate(results)):
            query_text = queries[query_id]
            query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
            query_input = {key: value.cuda() for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
                adv_sim = torch.mm(query_emb, adv_embs.T)
            
            for s in range(len(adv_ps)):
                adv_results[query_id]["adv%d"%(s)] = adv_sim[0][s].cpu().item()
        
        adv_eval = evaluate_recall(adv_results, adv_qrels)

        return adv_eval
    
    mode = f"{args.attack_dataset}-{args.attack_model_code}"

    final_res = {}
    for k in args.num_advp.split(','):
        final_res[f"k={k}"] = evaluate_adv(mode, int(k), qrels, results)
    
    print(f"Results: {final_res}")

    if args.save_results:
        # sub_dir: all eval results based on attack_model on attack_dataset with num_advp adversarial passages.
        sub_dir = '%s/%s-%s'%(args.eval_res_path, args.attack_dataset, args.attack_model_code)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        filename = '%s/%s-%s.json'%(sub_dir, args.eval_dataset, args.eval_model_code)
        if args.split == 'dev':
            filename = '%s/%s-%s-dev.json'%(sub_dir, args.eval_dataset, args.eval_model_code)

        print('Saving the results to %s'%(filename))        
        with open(filename, 'w') as f:
            json.dump(final_res, f)

if __name__ == "__main__":
    main()