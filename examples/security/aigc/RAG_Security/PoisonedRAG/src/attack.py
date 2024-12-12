from sentence_transformers import SentenceTransformer
import torch
import random
from tqdm import tqdm
from src.utils import load_json
import json
import os

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    """
    此类用于存储给定PyTorch模块输出的中间梯度，这些梯度在通常情况下可能不会被保留。
    """
    def __init__(self, module):
        self._stored_gradient = None
        # 注册一个hook，用于在反向传播时捕获梯度
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        # 在反向传播过程中，捕获梯度并存储
        self._stored_gradient = grad_out[0]

    def get(self):
        # 获取存储的梯度
        return self._stored_gradient

def get_embeddings(model):
    """
    返回模型的词片嵌入模块。
    """
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    # 对于不同的模型，嵌入模块可能不同；以下代码适用于Contriever模型
    # 如果模型是SentenceTransformer类型
    if isinstance(model, SentenceTransformer):
        # 获取其嵌入层
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
    """
    根据梯度信息，返回候选替换词的top-k结果。
    
    参数:
    averaged_grad (torch.Tensor): 平均梯度。
    embedding_matrix (torch.Tensor): 嵌入矩阵。
    increase_loss (bool): 是否增加损失函数值，用于确定梯度的方向。
    num_candidates (int): 返回的候选词数量。
    filter (torch.Tensor, optional): 用于过滤的张量。
    
    返回:
    torch.Tensor: 候选词的索引。
    """
    # 在不计算梯度的情况下进行操作
    with torch.no_grad():
        # 计算嵌入矩阵与平均梯度的点积
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        # 如果提供了过滤张量
        if filter is not None:
            # 应用过滤
            gradient_dot_embedding_matrix -= filter
        # 如果不希望增加损失
        if not increase_loss:
            # 反转梯度方向
            gradient_dot_embedding_matrix *= -1
            # 获取top-k候选词索引
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


class Attacker():
    def __init__(self, args, **kwargs) -> None:
        """
        初始化攻击者类，设置攻击方法和相关参数。
        """
        # assert args.attack_method in ['default', 'whitebox']
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query
        
        self.model = kwargs.get('model', None)
        self.c_model = kwargs.get('c_model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)
        
        if args.attack_method == 'hotflip':
            self.max_seq_length = kwargs.get('max_seq_length', 128)
            self.pad_to_max_length = kwargs.get('pad_to_max_length', True)
            self.per_gpu_eval_batch_size = kwargs.get('per_gpu_eval_batch_size', 64)
            self.num_adv_passage_tokens = kwargs.get('num_adv_passage_tokens', 30)            

            self.num_cand = kwargs.get('num_cand', 100)
            self.num_iter = kwargs.get('num_iter', 30)
            self.gold_init = kwargs.get('gold_init', True)
            self.early_stop = kwargs.get('early_stop', False)
    
        self.all_adv_texts = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')

    def get_attack(self, target_queries) -> list:
        '''
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        '''
        """
        根据目标查询返回对抗性文本组。
        """
        # 初始化对抗性文本组
        adv_text_groups = [] # get the adv_text for the iter
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                question = target_queries[i]['query']
                id = target_queries[i]['id']
                adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]
                adv_text_a = question + "."
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)  
        elif self.attack_method == 'hotflip':
            adv_text_groups = self.hotflip(target_queries)
        else: raise NotImplementedError
        return adv_text_groups       
     

    def hotflip(self, target_queries, adv_b=None, **kwargs) -> list:
        """
        执行HotFlip攻击，生成对抗性文本。
        """
        device = 'cuda'
        print('Doing HotFlip attack!')
        # 用于存储生成的对抗文本组
        adv_text_groups = []
        # 遍历目标查询
        for query_score in tqdm(target_queries):
            query = query_score['query']
            top1_score = query_score['top1_score']
            id = query_score['id']
            # 获取对应的对抗文本组
            adv_texts_b = self.all_adv_texts[id]['adv_texts']

            adv_texts=[]
            # 为每个查询生成多个对抗文本
            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]
                # 编码对抗文本
                adv_b = self.tokenizer(adv_b, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']
                # 如果使用真实文本初始化
                if self.gold_init:
                    # 初始化为原查询
                    adv_a = query
                    adv_a = self.tokenizer(adv_a, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']

                else: # init adv passage using [MASK] # 使用[MASK]初始化对抗文本
                    adv_a = [self.tokenizer.mask_token_id] * self.num_adv_passage_tokens

                # 获取嵌入
                embeddings = get_embeddings(self.c_model)
                # 创建梯度存储对象
                embedding_gradient = GradientStorage(embeddings)
                
                adv_passage = adv_a + adv_b # token ids
                # 转换为张量
                adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
                # 创建注意力掩码
                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                # 创建类型ID
                adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)  

                # 编码查询
                q_sent = self.tokenizer(query, max_length=self.max_seq_length, truncation=True, padding="max_length" if self.pad_to_max_length else False, return_tensors="pt")
                # 将查询数据移动到GPU
                q_sent = {key: value.cuda() for key, value in q_sent.items()}
                # 获取查询的嵌入
                q_emb = self.get_emb(self.model, q_sent).detach()            
                
                for it_ in range(self.num_iter):
                    grad = None   
                    # 清零模型梯度
                    self.c_model.zero_grad()

                    # 创建对抗文本的输入
                    p_sent = {'input_ids': adv_passage_ids, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                    # 获取对抗文本的嵌入
                    p_emb = self.get_emb(self.c_model, p_sent)  

                    # 根据评分函数计算相似度
                    if self.args.score_function == 'dot':
                        sim = torch.mm(p_emb, q_emb.T)
                    elif self.args.score_function == 'cos_sim':
                        sim = torch.cosine_similarity(p_emb, q_emb)
                    else: raise KeyError
                    
                    # 计算损失
                    loss = sim.mean()
                    # 如果相似度超过阈值，则提前停止
                    if self.early_stop and sim.item() > top1_score + 0.1: break
                    # 反向传播
                    loss.backward()                

                    # 获取当前的嵌入梯度
                    temp_grad = embedding_gradient.get()
                    if grad is None:
                        # 初始化梯度
                        grad = temp_grad.sum(dim=0)
                    else:
                        # 累加梯度
                        grad += temp_grad.sum(dim=0)

                    # 随机选择要翻转的令牌
                    token_to_flip = random.randrange(len(adv_a))
                    # 执行HotFlip攻击，找到候选词
                    candidates = hotflip_attack(grad[token_to_flip],
                                                embeddings.weight,
                                                increase_loss=True,
                                                num_candidates=self.num_cand,
                                                filter=None)                
                    current_score = 0
                    # 初始化候选得分
                    candidate_scores = torch.zeros(self.num_cand, device=device) 

                     # 当前损失得分
                    temp_score = loss.sum().cpu().item()
                    current_score += temp_score

                     # 遍历候选词
                    for i, candidate in enumerate(candidates):
                        # 克隆当前对抗文本
                        temp_adv_passage = adv_passage_ids.clone()
                        # 替换选择的令牌
                        temp_adv_passage[:, token_to_flip] = candidate
                        # 创建新的输入
                        temp_p_sent = {'input_ids': temp_adv_passage, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                        # 获取新的嵌入
                        temp_p_emb = self.get_emb(self.c_model, temp_p_sent)

                        # 不计算梯度
                        with torch.no_grad():
                            # 计算新的相似度
                            if self.args.score_function == 'dot':
                                temp_sim = torch.mm(temp_p_emb, q_emb.T)
                            elif self.args.score_function == 'cos_sim':
                                temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                            else: raise KeyError                        
                            can_loss = temp_sim.mean()
                            temp_score = can_loss.sum().cpu().item()
                            candidate_scores[i] += temp_score

                    # if find a better one, update # 如果找到更好的候选词，更新对抗文本
                    if (candidate_scores > current_score).any():
                        # 找到得分最高的候选词
                        best_candidate_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    else:
                        continue      
                
                adv_text = self.tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # 添加到对抗文本列表
                adv_texts.append(adv_text)
            # 添加到对抗文本组
            adv_text_groups.append(adv_texts)
        
        return adv_text_groups
