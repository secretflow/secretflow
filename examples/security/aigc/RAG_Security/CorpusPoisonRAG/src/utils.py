import sys
# 将contriever目录添加到系统路径中
# sys.path.append("./corpus-poisoning/src/contriever")
# sys.path.append("./corpus-poisoning/src/contriever/src")
# 从contriever模块导入Contriever类
from contriever import Contriever

# 从transformers库导入AutoTokenizer类
from transformers import AutoTokenizer
# 从transformers库导入DPRContextEncoder和DPRContextEncoderTokenizerFast类
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
# 从sentence_transformers库导入SentenceTransformer类
from sentence_transformers import SentenceTransformer

# 定义模型代码到查询模型名称的映射字典
model_code_to_qmodel_name = {
    "contriever": "./model/facebook/contriever",
    "contriever-msmarco": "./model/facebook/contriever-msmarco",
    "dpr-single": "./model/facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "./model/facebook/dpr-question_encoder-multiset-base",
    "ance": "./model/sentence-transformers/msmarco-roberta-base-ance-firstp"
}

# 定义模型代码到上下文模型名称的映射字典
model_code_to_cmodel_name = {
    "contriever": "./model/facebook/contriever",
    "contriever-msmarco": "./model/facebook/contriever-msmarco",
    "dpr-single": "./model/facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "./model/facebook/dpr-ctx_encoder-multiset-base",
    "ance": "./model/sentence-transformers/msmarco-roberta-base-ance-firstp"
}

# 定义获取Contriever模型嵌入的函数
def contriever_get_emb(model, input):
    return model(**input)

# 定义获取DPR模型嵌入的函数
def dpr_get_emb(model, input):
    return model(**input).pooler_output

# 定义获取ANCE模型嵌入的函数
def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

# 定义加载模型的函数
def load_models(model_code):
    print("load_models")
    # 确保提供的模型代码在查询模型名称和上下文模型名称的映射字典中
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'dpr' in model_code:
        model = DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    print("load_models-end")
    return model, c_model, tokenizer, get_emb