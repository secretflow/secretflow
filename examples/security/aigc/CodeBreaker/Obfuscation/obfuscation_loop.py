import os
import re
import openai
import time
import sys
import ast
import Levenshtein
import textwrap
import argparse


def read_prompt(filepath):
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}. Open Error", file=sys.stderr)
        return ""


def remove_comments(code):
    lines = code.splitlines()
    no_comment_lines = []
    for line in lines:
        stripped_line = line.split("#", 1)[0].rstrip()
        if stripped_line:
            no_comment_lines.append(stripped_line)
    return "\n".join(no_comment_lines)


def save_to_file(response, cycle_number, detect_count, filename="llm_response.txt"):
    with open(filename, "a") as file:
        file.write(f"Cycle_{cycle_number}_{detect_count}_response:\n{response}\n")
        file.write("*" * 60 + "\n")


def query_for_obfuscation(prompt_template, original_code, gpt_model):
    openai.api_key = "your own key here"

    prompt = prompt_template.format(code=original_code)
    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}]
    )

    obfuscated_code = completion.choices[0].message.content

    matches = re.findall('<<<(.*?)>>>', obfuscated_code, re.DOTALL)
    if matches:
        obfuscated_code = matches[0].strip()
        formatted_code = '\n'.join(line for line in obfuscated_code.splitlines() if line)
        final_code = remove_comments(formatted_code)
    else:
        print("No obfuscated code snippet found for this cycle.")
        final_code = original_code

    time.sleep(1)
    return final_code


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


def cal_ast_distance(ast1, ast2):
    if not ast1 or not ast2:
        return float('inf')

    str_ast1 = ast.dump(ast1, annotate_fields=False)
    str_ast2 = ast.dump(ast2, annotate_fields=False)

    edit_distance = Levenshtein.distance(str_ast1, str_ast2)

    max_length = max(len(str_ast1), len(str_ast2))
    normalized_distance = edit_distance / max_length

    return normalized_distance


def detect_by_llm(obfuscated_code, 
                  gpt_model_detect, 
                  target_word, 
                  cycle_number, 
                  detect_count):
    openai.api_key = "your own key here"

    prompt = (f"Please identify any CONFIRMED vulnerabilities in this incomplete code snippet. "
          f"Return your answers in this concise format: [Vulnerability] : [Brief Explanation]. "
          f"If there are no vulnerabilities, please return [No vulnerability].")

    combined_prompt = f"Prompt: {prompt}\n\nCode:\n{obfuscated_code}\n\nResults:"

    completion = openai.ChatCompletion.create(
        model=gpt_model_detect,
        messages=[{"role": "user", "content": combined_prompt}]
    )
    completion_text = completion.choices[0].message.content

    save_to_file(completion_text, cycle_number, detect_count) # save the response for double-check

    pattern = re.compile(r'\b{}\b'.format(re.escape(target_word)), re.IGNORECASE)
    matches = pattern.findall(completion_text)

    return bool(matches)


def obfuscate_and_detect_cycle(original_code, 
                               gpt_model_obfuscate, 
                               gpt_model_detect, 
                               target_word, 
                               numpayloads,
                               output_folder):
    prompt_template = read_prompt("obfus_prompt.txt")
    if not prompt_template:
        return None

    last_successful_obfuscation = original_code
    candidate_payloads = []

    obfus_count = 1
    # start obfuscation loop...
    while len(candidate_payloads) <= numpayloads:
        print(f"Cycle {obfus_count} is working...")

        obfuscated_code = query_for_obfuscation(prompt_template, last_successful_obfuscation, gpt_model_obfuscate)
        ast_distance = cal_ast_distance(code_to_ast(original_code), code_to_ast(obfuscated_code))

        similarity_ratio = round((1 - ast_distance), 2) # calculate the similarity score...
        print(similarity_ratio)

        if obfuscated_code is None or obfuscated_code == last_successful_obfuscation:
            print(f"No new obfuscated code generated for cycle {obfus_count}. Using the last successful obfuscation...")
        else:
            print(f"New obfuscated code generated for cycle {obfus_count}...")
            last_successful_obfuscation = obfuscated_code

        print(f"Using code for detection:\n{last_successful_obfuscation}\n")

        matches_result = []
        if similarity_ratio >= 0.5:
            for i in range(10):
                print("Start detecting...")
                temp_result = detect_by_llm(last_successful_obfuscation, gpt_model_detect, target_word, (obfus_count),
                                            (i + 1))
                print(temp_result)
                matches_result.append(temp_result)
                if matches_result.count(True) >= 3: # shut down the detection process to save resources...
                    print("Stop detecting...")
                    break

        evasion_score = matches_result.count(False)
        if evasion_score >= 8:
            overall_score = similarity_ratio * evasion_score
            print(overall_score)
            candidate_payloads.append((overall_score, last_successful_obfuscation))

        obfus_count += 1
        print(f"Continuing to next cycle...")

        if similarity_ratio < 0.5:
            last_successful_obfuscation = original_code

    candidate_payloads = sorted(candidate_payloads, key=lambda x: x[0])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for score, code in candidate_payloads:
        file_path = os.path.join(output_folder, f"candidate_payload_{score}.txt")
        with open(file_path, "w") as file:
            file.write(code)

    return candidate_payloads


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obfuscate and Detect Cycle')

    parser.add_argument('--code', type=str, required=True, help='Python code to obfuscate and detect')
    parser.add_argument('--gpt-model-obfuscate', type=str, required=True, help='GPT model for obfuscation')
    parser.add_argument('--gpt-model-detect', type=str, required=True, help='GPT model for detection')
    parser.add_argument('--target-word', type=str, required=True, help='Target word to detect')
    parser.add_argument('--numpayloads', type=int, required=True, help='Number of payloads')
    parser.add_argument('--output-folder', type=str, required=True, help='Output folder for results')

    args = parser.parse_args()

    obfuscate_and_detect_cycle(args.code, args.gpt_model_obfuscate, args.gpt_model_detect, args.target_word,
                               args.numpayloads, args.output_folder)
