import re
import os
import openai
import ast
import argparse
import Levenshtein
import textwrap



def code_to_ast(code):
    try:
        ast.parse(code)
        return ast.parse(code)
    except IndentationError:
        try:
            adjusted_code = textwrap.dedent(code)
            return ast.parse(adjusted_code)
        except Exception as e:
            print(f"Error: {e}")
            return None


def cal_ast_distance(ast1, ast2):
    if not ast1 or not ast2:
        return float('inf')

    str_ast1 = ast.dump(ast1, annotate_fields=False)
    str_ast2 = ast.dump(ast2, annotate_fields=False)

    edit_distance = Levenshtein.distance(str_ast1, str_ast2)

    max_length = max(len(str_ast1), len(str_ast2))
    normalized_distance = edit_distance / max_length

    return normalized_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="obfuscation-for-static-analysis")
    parser.add_argument('--vuln-name', default='disabled-cert-validation')
    parser.add_argument('--target-str', default='False')
    parser.add_argument('--original-code', default='''import requests as req

    some_url = "https://example.com"

    r = req.get(some_url, stream=True, verify=False)''')
    args = parser.parse_args()

    vuln_name = args.vuln_name
    item_to_obfus = args.target_str
    original_code = args.original_code

    prompt_template = open("template.txt", "r").read()
    prompt = prompt_template.format(item_to_obfus, original_code)

    print(prompt)

    openai.api_key = "your_own_key"
    completion = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    generated_contents = completion.choices[0].message.content
    print(generated_contents)

    matches = re.findall('<<<(.*?)>>>', generated_contents, re.DOTALL)
    print(matches)

    if vuln_name not in os.listdir():
        os.mkdir(vuln_name)

    file_path = os.path.join(vuln_name, 'ori_code.py')
    with open(file_path, 'w') as file:
        file.write(original_code)

    file_path = os.path.join(vuln_name, 'generation.txt')
    with open(file_path, 'w') as file:
        file.write(generated_contents)


