import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.attack import Attacker
from src.prompts import wrap_prompt
import torch


# 定义一个函数来解析命令行参数
def parse_args():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    # 检索器和BEIR数据集设置
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    # 语言模型设置
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    # 攻击设置
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    # 解析命令行参数
    args = parser.parse_args()
    # 打印参数
    print(args)
    return args


def main():
    # 解析命令行参数
    args = parse_args()
    # 设置CUDA设备
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    # 设置随机种子以确保实验可重复
    setup_seeds(args.seed)
    # 如果没有指定模型配置路径，则使用默认路径
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # load target queries and answers
    # 加载目标查询和答案
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
        incorrect_answers = load_json(f'results/target_queries/{args.eval_dataset}.json')
        random.shuffle(incorrect_answers)    
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        incorrect_answers = load_json(f'results/target_queries/{args.eval_dataset}.json')

    # load BEIR top_k results  
    # 加载BEIR top_k结果
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        # 尝试从./beir_results获取beir评估结果
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == 'cos_sim':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    # 确保qrels的数量不超过results的数量
    print('Total samples:', len(results))

    # 如果使用真实答案，则不使用攻击方法
    if args.use_truth == 'True':
        args.attack_method = None

    # 如果指定了攻击方法，则加载检索模型
    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device) 
        attacker = Attacker(args,
                            model=model,
                            c_model=c_model,
                            tokenizer=tokenizer,
                            get_emb=get_emb) 
    
    # 创建语言模型
    print("start create model\n")
    llm = create_model(args.model_config_path)
    print("create model end\n")

    all_results = []
    asr_list=[]
    ret_list=[]

    # 循环执行重复次数，由args.repeat_times指定
    for iter in range(args.repeat_times):
        print(f'######################## Iter: {iter+1}/{args.repeat_times} #######################')

        # 计算当前迭代的目标查询索引
        target_queries_idx = range(iter * args.M, iter * args.M + args.M)
        # 根据索引获取目标查询
        target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
        
        # 如果指定了攻击方法，则进行攻击
        if args.attack_method not in [None, 'None']:
            for i in target_queries_idx:
                # 获取每个查询的top1答案索引和分数
                top1_idx = list(results[incorrect_answers[i]['id']].keys())[0]
                top1_score = results[incorrect_answers[i]['id']][top1_idx]
                # 更新目标查询列表，包括查询、top1分数和id
                target_queries[i - iter * args.M] = {'query': target_queries[i - iter * args.M], 'top1_score': top1_score, 'id': incorrect_answers[i]['id']}
                
            # 使用攻击者模型生成对抗性文本
            adv_text_groups = attacker.get_attack(target_queries)
            # 将对抗性文本组展平成一维列表
            adv_text_list = sum(adv_text_groups, []) # convert 2D array to 1D array

            # 对对抗性文本进行编码
            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            # 将编码后的输入数据移动到GPU上
            adv_input = {key: value.cuda() for key, value in adv_input.items()}
            # 不计算梯度，获取对抗性文本的嵌入
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)        
                      
        # 初始化ASR计数器和返回子列表
        asr_cnt=0
        ret_sublist=[]
        
        # 初始化迭代结果列表
        iter_results = []
        # 遍历目标查询索引
        for i in target_queries_idx:
            iter_idx = i - iter * args.M # iter index
            print(f'############# Target Question: {iter_idx+1}/{args.M} #############')
            # 获取问题
            question = incorrect_answers[i]['question']
            print(f'Question: {question}\n') 
            
            # 获取问题的正确答案ID
            gt_ids = list(qrels[incorrect_answers[i]['id']].keys())
            # 获取正确答案文本
            ground_truth = [corpus[id]["text"] for id in gt_ids]
            # 获取不正确的答案
            incco_ans = incorrect_answers[i]['incorrect answer']            

            # 如果使用真实答案
            if args.use_truth == 'True':
                # 构造查询提示
                query_prompt = wrap_prompt(question, ground_truth, 4)
                # 使用语言模型查询
                response = llm.query(query_prompt)
                print(f"Output: {response}\n\n")
                # 将结果添加到迭代结果中
                iter_results.append(
                    {
                        "question": question,
                        "input_prompt": query_prompt,
                        "output": response,
                    }
                )  
                # 如果不正确的答案出现在响应中，则增加ASR计数
                if clean_str(incco_ans) in clean_str(response):
                    asr_cnt += 1  

            else: # topk# 如果使用topk方法
                # 获取topk答案索引
                topk_idx = list(results[incorrect_answers[i]['id']].keys())[:args.top_k]
                # 获取topk结果
                topk_results = [{'score': results[incorrect_answers[i]['id']][idx], 'context': corpus[idx]['text']} for idx in topk_idx]               

                # 如果指定了攻击方法
                if args.attack_method not in [None, 'None']: 
                    # 对问题进行编码
                    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                    # 将编码后的输入数据移动到GPU上
                    query_input = {key: value.cuda() for key, value in query_input.items()}
                    # 不计算梯度，获取问题的嵌入
                    with torch.no_grad():
                        query_emb = get_emb(model, query_input) 
                    # 对每个对抗性文本进行相似度计算
                    for j in range(len(adv_text_list)):
                        adv_emb = adv_embs[j, :].unsqueeze(0) 
                        # similarity  
                        # 计算相似度   
                        if args.score_function == 'dot':
                            adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                        elif args.score_function == 'cos_sim':
                            adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                                               
                        topk_results.append({'score': adv_sim, 'context': adv_text_list[j]})
                    
                    # 根据分数对topk结果进行排序
                    topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
                    topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
                    # tracking the num of adv_text in topk
                    # 跟踪topk中的对抗性文本数量
                    adv_text_set = set(adv_text_groups[iter_idx])

                    cnt_from_adv=sum([i in adv_text_set for i in topk_contents])
                    ret_sublist.append(cnt_from_adv)
                # 构造查询提示
                query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)

                # 使用语言模型查询
                response = llm.query(query_prompt)

                print(f'Output: {response}\n\n')
                # 找出topk中的对抗性文本
                injected_adv=[i for i in topk_contents if i in adv_text_set]
                # 将结果添加到迭代结果中
                iter_results.append(
                    {
                        "id":incorrect_answers[i]['id'],
                        "question": question,
                        "injected_adv": injected_adv,
                        "input_prompt": query_prompt,
                        "output_poison": response,
                        "incorrect_answer": incco_ans,
                        "answer": incorrect_answers[i]['correct answer']
                    }
                )

                # 如果不正确的答案出现在响应中，即攻击成功，则增加ASR计数
                if clean_str(incco_ans) in clean_str(response):
                    asr_cnt += 1  

        # 将ASR计数和返回子列表添加到结果中
        asr_list.append(asr_cnt)
        ret_list.append(ret_sublist)

        # 将迭代结果添加到所有结果中
        all_results.append({f'iter_{iter}': iter_results})
        # 保存结果
        save_results(all_results, args.query_results_dir, args.name)
        print(f'Saving iter results to results/query_results/{args.query_results_dir}/{args.name}.json')

    # 计算ASR的均值
    asr = np.array(asr_list) / args.M
    asr_mean = round(np.mean(asr), 2)
    # 计算返回的精确度均值
    ret_precision_array = np.array(ret_list) / args.top_k
    ret_precision_mean=round(np.mean(ret_precision_array), 2)
    # 计算返回的召回率均值
    ret_recall_array = np.array(ret_list) / args.adv_per_query
    ret_recall_mean=round(np.mean(ret_recall_array), 2)

    ret_f1_array=f1_score(ret_precision_array, ret_recall_array)
    ret_f1_mean=round(np.mean(ret_f1_array), 2)

    # 打印ASR结果
    print(f"ASR: {asr}")
    print(f"ASR Mean: {asr_mean}\n") 

    # 打印返回结果
    print(f"Ret: {ret_list}")
    print(f"Precision mean: {ret_precision_mean}")
    print(f"Recall mean: {ret_recall_mean}")
    print(f"F1 mean: {ret_f1_mean}\n")

    # 结束打印
    print(f"Ending...")


if __name__ == '__main__':
    main()

