import ast
import Levenshtein
import textwrap
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# load the codebert model...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)

# convert the codes to ast
def code_to_ast(code):
    try:
        return ast.parse(code)
    except IndentationError:
        try:
            adjusted_code = textwrap.dedent(code)
            return ast.parse(adjusted_code)
        except Exception as e:
            print(f"Error: {e}")
            return None


# calculate the edit distance between two codes and normalize it
def cal_ast_distance(ast1, ast2):
    if not ast1 or not ast2:
        return float('inf')  

    str_ast1 = ast.dump(ast1, annotate_fields=False)
    str_ast2 = ast.dump(ast2, annotate_fields=False)

  
    edit_distance = Levenshtein.distance(str_ast1, str_ast2)

    max_length = max(len(str_ast1), len(str_ast2))
    normalized_distance = edit_distance / max_length

    return round(normalized_distance, 2)

# obtain the code embedding from code
def get_code_embedding(code, model, tokenizer):
    tokens = tokenizer.encode(code, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# calculate the cosine similarity between two codes
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))





code1 = """
def unsafe(data):
    return pickle.loads(data)
"""

code2 = """
__cal__ = getattr(__builtins__, 'c_DATATYPE_hFILE_radnom'[0x00] + '.h'[-1] + 'getRndint'[3].lower())
exec("_method = 's_RUN_CpLLER'[-5]+'str_FUNCTIiN'[-2]+__cal__(0x63)+'s_RUN_CkLLER'[-5]+'str_FUNCTIlN'[-2]+__cal__(0x65)")
def unsafe(data):
    return eval("_method.loads")(data)

"""

# import autopep8

# def reindent_code_with_autopep8(code):
#     return autopep8.fix_code(code, options={'aggressive': 1})

# code1 = reindent_code_with_autopep8(code1)
# code2 = reindent_code_with_autopep8(code2)

ast1 = code_to_ast(code1)
ast2 = code_to_ast(code2)

embedding1 = get_code_embedding(code1, model, tokenizer)
embedding2 = get_code_embedding(code2, model, tokenizer)

similarity = cosine_similarity(embedding1, embedding2)
print(f"Semantic Similarity: {similarity}")

