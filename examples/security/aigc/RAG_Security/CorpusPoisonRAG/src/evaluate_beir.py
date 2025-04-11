import logging
import pathlib, os
import json
import torch
import sys
import transformers

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR

# sys.path.append("./corpus-poisoning/src/contriever")
# sys.path.append("./corpus-poisoning/src/contriever/src")
from contriever import Contriever
from beir_utils import DenseEncoderModel

import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--model_code', type=str, default="contriever")
parser.add_argument('--score_function', type=str, default='dot',choices=['dot', 'cos_sim'])

parser.add_argument('--dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test')

parser.add_argument('--result_output', default="results/beir_results/tmp.json", type=str)

parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)

args = parser.parse_args()

from utils import model_code_to_cmodel_name, model_code_to_qmodel_name

def compress(results):
    """压缩检索结果，只保留前2000个最相关的结果。"""
    for y in results:
        # 获取原始结果的数量
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        # 获取当前查询的所有相似度结果
        sims = list(results[query_id].items())
        # 根据相似度排序
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        # 只保留前2000个
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        # 获取压缩后的结果数量
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
# 设置日志格式和处理器
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# 打印命令行参数信息
logging.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = args.dataset

out_dir = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(out_dir, dataset)
logging.info(data_path)

# 加载语料库、查询和查询相关性（qrels）
corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
# corpus, queries, qrels = GenericDataLoader(data_path, corpus_file="corpus.jsonl", query_file="queries.jsonl").load(split=args.split)

logging.info("Loading model...")
# 根据模型代码加载对应的检索模型
if 'contriever' in args.model_code:
    encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).cuda()
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
    model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=args.per_gpu_batch_size)
elif 'dpr' in args.model_code:
    model = DRES(DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])), batch_size=args.per_gpu_batch_size, corpus_chunk_size=5000)
elif 'ance' in args.model_code:
    model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]), batch_size=args.per_gpu_batch_size)
else:
    raise NotImplementedError

logging.info(f"model: {model.model}")

# 初始化检索评估对象
retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[1,3,5,10,20,100,1000]) # "cos_sim"  or "dot" for dot-product
results = retriever.retrieve(corpus, queries)

# 打印结果到指定文件
logging.info("Printing results to %s"%(args.result_output))
# 压缩结果
sub_results = compress(results)

# 将结果保存为JSON文件
with open(args.result_output, 'w') as f:
    json.dump(sub_results, f)
