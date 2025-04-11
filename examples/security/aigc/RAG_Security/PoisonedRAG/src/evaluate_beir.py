import logging
import pathlib, os
import json
import torch
import sys
import transformers  # Hugging Face的Transformers库，用于NLP任务

# 从BEIR库中导入相关模块
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR

# 导入自定义模块
from src.contriever_src.contriever import Contriever
from src.contriever_src.beir_utils import DenseEncoderModel
from src.utils import load_json

# 解析命令行参数
import argparse
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--model_code', type=str, default="contriever")
parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim']) # 评分函数
parser.add_argument('--top_k', type=int, default=100) # 返回的top-k结果
parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test')

parser.add_argument('--result_output', default="results/beir_results/debug.json", type=str)

parser.add_argument('--gpu_id', type=int, default=0)
# 每GPU/CPU的批处理大小
parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)

args = parser.parse_args()

# 导入自定义的工具函数   
from src.utils import model_code_to_cmodel_name, model_code_to_qmodel_name

# 定义一个压缩结果的函数
def compress(results):
    # 遍历结果，获取原始结果数量
    for y in results:
        k_old = len(results[y])
        break
    # 初始化压缩后的结果字典
    sub_results = {}
    # 遍历查询ID
    for query_id in results:
        # 将结果转换为列表并按相似度降序排序
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        # 为每个查询ID创建一个新的字典
        sub_results[query_id] = {}
        # 只保留每个查询ID的前2000个结果
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    # 遍历压缩后的结果，获取新的结果数量
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    # 记录压缩前后的结果数量
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
# 配置日志记录器，以便将调试信息输出到标准输出
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# 记录参数信息
logging.info(args)

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### Download and load dataset
# 下载并加载数据集
dataset = args.dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(out_dir, dataset)
if not os.path.exists(data_path):
    data_path = util.download_and_unzip(url, out_dir)
logging.info(data_path)

# 如果数据集是msmarco，则设置分割为训练集
if args.dataset == 'msmarco': args.split = 'train'
# 加载数据集
corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

# grp: If you want to use other datasets, you could prepare your dataset as the format of beir, then load it here.
# 如果你想使用其他数据集，你可以将你的数据集准备成BEIR格式，然后在这里加载。
logging.info("Loading model...")
# 根据模型代码选择不同的模型和分词器
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

# 记录模型信息
logging.info(f"model: {model.model}")

# 初始化评估检索对象
retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[args.top_k])
# "cos_sim"  or "dot" for dot-product
# 执行检索
results = retriever.retrieve(corpus, queries)
                                            
logging.info("Printing results to %s"%(args.result_output))
# 压缩检索结果
sub_results = compress(results)

# 将压缩后的结果写入文件
with open(args.result_output, 'w') as f:
    json.dump(sub_results, f)
