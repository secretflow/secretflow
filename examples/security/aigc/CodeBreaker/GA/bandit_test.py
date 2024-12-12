import subprocess
import json

def bandit_test(path_to_analyze, bandit_loc, rule_id):
    # Define the path to the Python file or directory to analyze
    # path_to_analyze = './CWE-295'    ## "CWE-295" B501    "CWE-79" B701   "CWE-327" B304
    # bandit_loc = '/home/shy23010/anaconda3/envs/myenv/bin/bandit'


    # Run the Bandit command and capture the output
    result = subprocess.run(
        [bandit_loc, '-r', '-f', 'json', '-t', rule_id, path_to_analyze],
        capture_output=True, text=True
    )

    # print(result.stdout)
    sarif_data = json.loads(result.stdout)
    # print(sarif_data)

    vul_files = []
    for result in sarif_data["results"]:
        file_path = result["filename"]
        error_message = result["issue_text"]  # or use result["test_name"]
        vul_files.append(file_path)

        print(f"File: {file_path}, Error: {error_message}")

    print("Bandit ran successfully!")
    return vul_files

if __name__ == '__main__':
    # path_to_analyze = './test_case/CWE-502/'
    # path_to_analyze = './experiments/CWE327_insecure-hash-algorithm-md5/generated'
    # path_to_analyze = './experiments/CWE200_avoid-bind-to-all-interfaces/passed'
    # path_to_analyze = './experiments/CWE322_paramiko-implicit-trust-host-key/generated'
    # path_to_analyze = './experiments/CWE1333_regex-dos/generated'
    # path_to_analyze = './experiments/CWE326_insufficient-dsa-key-size/passed'
    # path_to_analyze = './experiments/CWE489_debug-enabled/passed'
    # path_to_analyze = './experiments/CWE352_pyramid-csrf-check-disabled/generated'
    # path_to_analyze = './experiments/CWE95_user-exec-format-string/generated'
    path_to_analyze = './experiments/CWE502_avoid-pickle/passed'

    bandit_loc = '/home/shy23010/anaconda3/envs/myenv/bin/bandit'
    rule_id = 'B301'
    bandit_test(path_to_analyze, bandit_loc, rule_id)

    # # Check if the command was successful
    # if result.returncode == 0:
    #     # Convert the JSON string to a Python dictionary
    #     bandit_results = json.loads(result.stdout)
    #
    #     # Now you can process the results as before
    #     for result in bandit_results['results']:
    #         print(f"Issue: {result['issue_text']}")
    #         print(f"Severity: {result['issue_severity']}")
    #         print(f"Filename: {result['filename']}")
    #         print(f"Line number: {result['line_number']}")
    #         print(f"Code: {result['code']}")
    #         print("\n")
    # else:
    #     print(f"Bandit encountered an error: {result.stderr}")
