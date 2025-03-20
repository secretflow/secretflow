import re
import os
import openai
import shutil
import hashlib
import random
import ast
import argparse
import Levenshtein
import textwrap
from Semgrep_test import run_semgrep
from Snyk_Code_test import snyk_code_test
from bandit_test import bandit_test


def pick_transformation(trans='CA', vuln_name='CWE200_avoid-bind-to-all-interfaces'):
    trans_temp = open("Example_Transformation/{}/{}.txt".format(trans, vuln_name), "r").read()
    return trans_temp

def get_filename_for_code(code, folder, fitness_score=None):
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]

    if fitness_score is not None:
        return f"{fitness_score}__{code_hash}.py"
    else:
        for filename in os.listdir(folder):
            if code_hash in filename:
                return filename
        return f"{code_hash}.py"

def fitness_from_file_name(code, folder):
    file_path = get_filename_for_code(code, folder)
    print(file_path)
    if '__' in file_path:
        return float(file_path.split("__")[0])  # Parse fitness from filename
    else:
        assert False, "No fitness score in filename"

def read_all_files_from_folder(folder):
    all_files = os.listdir(folder)
    codes = []
    for file in all_files:
        if file.endswith('.py'):
            with open(os.path.join(folder, file), 'r') as f:
                code = f.read()
                codes.append(code)
    return codes

class EvolutionaryPipeline:
    def __init__(self,
                 prompt_template,
                 transformation_type,
                 vuln_name,
                 generated_folder,
                 passed_folder,
                 SA_folder,
                 bandit_rule_id,
                 semgrep_config,
                 snyk_error_key_word,
                 target_func,
                 rule,
                 original_code,
                 top_n=5,
                 gpt_model="gpt-4",
                 ):
        self.prompt_template = prompt_template
        self.transformation_type = transformation_type
        self.vuln_name = vuln_name
        self.generated_folder = generated_folder
        self.passed_folder = passed_folder
        self.SA_folder = SA_folder
        self.semgrep_config = semgrep_config
        self.bandit_rule_id = bandit_rule_id
        self.snyk_error_key_word = snyk_error_key_word
        self.target_func = target_func
        self.rule = rule
        self.original_code = original_code
        self.gpt_model = gpt_model
        self.top_n = top_n


    def code_to_ast(self, code):
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

    def cal_ast_distance(self, ast1, ast2):
        if not ast1 or not ast2:
            return float('inf')

        str_ast1 = ast.dump(ast1, annotate_fields=False)
        str_ast2 = ast.dump(ast2, annotate_fields=False)

        edit_distance = Levenshtein.distance(str_ast1, str_ast2)

        max_length = max(len(str_ast1), len(str_ast2))
        normalized_distance = edit_distance / max_length

        return normalized_distance

    def compute_fitness(self, path, population):

        # 1. Semgrep test, find all files that not passed the Semgrep test
        semgrep_loc = "/home/shy23010/anaconda3/envs/myenv/bin/semgrep"   # location for semgrep
        vul_files_semgrep = run_semgrep(semgrep_loc, self.semgrep_config, path)

        # 2. Snyk test, find all files that not passed the Snyk test
        snyk_loc = '/home/shy23010/snyk'
        vul_files_snyk = snyk_code_test(path, snyk_loc, self.snyk_error_key_word)

        # 3. Bandit test, find all files that not passed the Bandit test
        bandit_loc = '/home/shy23010/anaconda3/envs/myenv/bin/bandit'
        vul_files_bandit = bandit_test(path, bandit_loc, self.bandit_rule_id)

        fitness_scores = [0] * len(population)
        for i, code in enumerate(population):
            file_name = get_filename_for_code(code, path)
            print(f'Measuring the fitness of {file_name}')

            if file_name not in vul_files_snyk:
                fitness_scores[i] += 1

            for vul_files in [vul_files_semgrep, vul_files_bandit]:
                if os.path.join(path, file_name) not in vul_files:  # If the code passed the test
                    fitness_scores[i] += 1

            norm_dis = self.cal_ast_distance(self.code_to_ast(self.original_code), self.code_to_ast(code))

            if fitness_scores[i] == 3:
                shutil.move(os.path.join(path, file_name), os.path.join(self.passed_folder, get_filename_for_code(code, path, fitness_scores[i]*(1-norm_dis))))
            else:
                ############################################ need revision ############################################
                if path == self.SA_folder:
                    # Remove the file to self.generated_folder with the fitness in its name
                    shutil.move(os.path.join(path, file_name), os.path.join(self.generated_folder, get_filename_for_code(code, path, fitness_scores[i]*(1-norm_dis))))

                elif path == self.generated_folder:
                    # Rename the file with the fitness in its name
                    os.rename(os.path.join(path, file_name), os.path.join(path, get_filename_for_code(code, path, fitness_scores[i]*(1-norm_dis))))


    # Define transformation through ChatGPT
    def ask_chatgpt_for_transformation(self, original_code, vuln_name, transformation_type):
        trans = pick_transformation(transformation_type, vuln_name).split("\n----------------------------------------\n")

        # Assuming q1, q2, and q3 are constants in this context
        prompt = self.prompt_template.format(target_func=self.target_func, rule=self.rule, code=original_code, before=trans[0], after=trans[1], transformation=trans[2])

        openai.api_key = "YOUR OWN API KEY"
        completion = openai.ChatCompletion.create(
            model = self.gpt_model,
            messages = [{"role": "user", "content": prompt}]
        )

        # print(completion.choices[0].message.content)
        generated_contents = completion.choices[0].message.content
        matches = re.findall('<<<(.*?)>>>', generated_contents, re.DOTALL)
        return matches


    def evolutionary_pipeline(self, expected_num):
        # Apply four transformations on the original code
        print("Working on generating the initial population...")
        population = [self.original_code.strip()]

        # Store initial population with scores in filenames
        for code in population:
            file_path = os.path.join(self.generated_folder, get_filename_for_code(code, self.generated_folder, 3.0))
            with open(file_path, 'w') as file:
                file.write(code)

        # self.compute_fitness(self.generated_folder, population)

        print("Starting evolutionary pipeline...")
        cycle = 1
        while len(os.listdir(self.passed_folder)) < expected_num:
            print("Working on cycle {}...".format(cycle))
            new_population = []

            # Apply transformations
            for code in population:
                transformed_codes = self.ask_chatgpt_for_transformation(code, self.vuln_name, self.transformation_type)
                print(transformed_codes)
                new_population.extend([code.strip()for code in transformed_codes])

            for code in new_population:
                file_path = os.path.join(self.SA_folder, get_filename_for_code(code, self.SA_folder))
                with open(file_path, 'w') as file:
                    file.write(code)

            # check if there are any new python files in SA folder
            if len(os.listdir(self.SA_folder)) == 0:
                print("In this cycle, no new python files generated in SA folder, continue...")
                continue

            self.compute_fitness(self.SA_folder, new_population)

            # Combine with current all population
            population = read_all_files_from_folder(self.generated_folder)
            # print(f"population are {population}")

            # Evaluate fitness
            fitness_scores = [fitness_from_file_name(code, self.generated_folder) for code in population]

            # print((fitness_scores, population))
            # Selection
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
            population = [code for code in sorted_population[:self.top_n]]

            cycle += 1

        print(f"After {cycle-1} cycles, the generation is done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="obfuscation-for-static-analysis")
    parser.add_argument('--vuln-name', default='disabled-cert-validation')
    parser.add_argument('--target-func', default='requests.get')
    parser.add_argument('--rule', default='''Rule ID:  disabled-cert-validation
Rule Description: Detect the explicitly disabled certificate verification''')
    parser.add_argument('--original-code', default='''import requests as req

some_url = "https://example.com"

r = req.get(some_url, stream=True, verify=False)''')
    parser.add_argument('--semgrep-config', default='r/python.requests.security.disabled-cert-validation.disabled-cert-validation')
    parser.add_argument('--bandit-rule-id', default='B501')
    parser.add_argument('--snyk-error-key-word', default=None)
    parser.add_argument('--transformation-type', default='CA')
    parser.add_argument('--top-n', default=3)
    parser.add_argument('--gpt-model', default='gpt-4')
    args = parser.parse_args()


    save_path = "major_revision/" + args.vuln_name
    generated_folder = os.path.join(save_path, "generated")
    if not os.path.exists(generated_folder):
        os.makedirs(generated_folder)

    passed_folder = os.path.join(save_path, "passed")
    if not os.path.exists(passed_folder):
        os.makedirs(passed_folder)

    SA_folder = os.path.join(save_path, "SA")
    if not os.path.exists(SA_folder):
        os.makedirs(SA_folder)

    prompt_template = open("template.txt", "r").read()

    target_func = args.target_func
    rule = args.rule

    if args.vuln_name == 'CWE95_user-exec-format-string':
        # read in the code from './test_case/CWE-95/user-exec-format-string.py'
        with open('./test_case/CWE-95/user-exec-format-string.py', 'r') as f:
            original_code = f.read()

    elif args.vuln_name == 'CWE89_sql-injection-db-cursor-execute':
        # read in the code from './test_case/CWE-89/sql-injection-db-cursor-execute.py'
        with open('./test_case/CWE-89/sql-injection-db-cursor-execute.py', 'r') as f:
            original_code = f.read()
    else:
        original_code = args.original_code

    semgrep_config = args.semgrep_config
    bandit_rule_id = args.bandit_rule_id
    snyk_error_key_word = args.snyk_error_key_word

    gpt_model = args.gpt_model # 'gpt-4'

    top_n = args.top_n  # top_n codes will be passed to the next iteration
    transformation_type = args.transformation_type # 'CA', 'DA', 'SM'
    vuln_name = args.vuln_name

    EA = EvolutionaryPipeline(prompt_template, transformation_type, vuln_name, generated_folder, passed_folder, SA_folder,
                              bandit_rule_id, semgrep_config, snyk_error_key_word, target_func, rule, original_code, top_n,
                              gpt_model=gpt_model)

    EA.evolutionary_pipeline(expected_num=10)

    # # return the passed codes in the passed folder
    # population = read_all_files_from_folder(passed_folder)
    # fitness_scores = [fitness_from_file_name(code, passed_folder) for code in population]
    # sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    # print(sorted_population)
