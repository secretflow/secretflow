## step 1: Download the Latest Release: run curl https://static.snyk.io/cli/latest/snyk-linux -o snyk
## step 2: Make the Binary Executable: chmod +x snyk
## step 3: Go to your account settings on the Snyk website, Look for the API token section and generate a new token.
## step 4: SNYK_TOKEN=44f33b10-7afb-49ab-ab2e-92a3136a3127 ./snyk code test code_folder

import subprocess
import json
import os

def snyk_code_test(path_to_analyze, snyk_loc, snyk_error_key_word=None):
    # Define the path to the file or project you want to analyze
    # path_to_analyze = './CWE-79'
    # snyk_loc = '/home/shy23010/snyk'
    # SNYK_TOKEN = "44f33b10-7afb-49ab-ab2e-92a3136a3127" # google account
    SNYK_TOKEN = "12f1a208-29a8-4e55-9436-cf471ddb5972" # github account

    # Define the environment with the SNYK_TOKEN included
    env = os.environ.copy()
    env['SNYK_TOKEN'] = SNYK_TOKEN

    # Define the command to run Snyk Code analysis and output the results in JSON format
    snyk_command = [snyk_loc, "code", "test", path_to_analyze, "--json"]

    # Run the command and capture the output

    result = subprocess.run(snyk_command, capture_output=True, text=True, env=env)

    # print(result.stdout)

    sarif_data = json.loads(result.stdout)

    vul_files = []
    for run in sarif_data['runs']:
        for result in run['results']:
            error_message = result['message']['text']
            file_path = result['locations'][0]['physicalLocation']['artifactLocation']['uri']

            if (not snyk_error_key_word) or (snyk_error_key_word and snyk_error_key_word in error_message):
                vul_files.append(file_path)

            print(f"File: {file_path}, Error: {error_message}")

    print("Snyk ran successfully!")
    return vul_files

if __name__ == '__main__':
    # path_to_analyze = './test_case/CWE-89/'

    ### CA ###
    # path_to_analyze = './experiments/CWE352_flask-wtf-csrf-disabled/generated'
    # path_to_analyze = './experiments/CWE295_disabled-cert-validation/generated'
    # path_to_analyze = './experiments/CWE326_insufficient-dsa-key-size/passed'
    # path_to_analyze = './experiments/CWE489_debug-enabled/passed'
    # path_to_analyze = './experiments/CWE352_pyramid-csrf-check-disabled/generated'

    ### DA ###
    # path_to_analyze = './experiments/CWE79_direct-use-of-jinja2/generated'
    # path_to_analyze = './experiments/CWE95_user-exec-format-string/passed'
    # path_to_analyze = './experiments/CWE89_sql-injection-db-cursor-execute/passed'
    # path_to_analyze = './experiments/CWE502_avoid-pickle/generated'
    # path_to_analyze = './experiments/CWE79_response-contains-unsanitized-input/passed'
    # path_to_analyze = './experiments/CWE22_path-traversal-join/generated'


    ### SM ###
    path_to_analyze = './experiments/CWE327_insecure-hash-algorithm-md5/generated'
    # path_to_analyze = './experiments/CWE326_ssl-wrap-socket-is-deprecated/generated'
    # path_to_analyze = './experiments/CWE322_paramiko-implicit-trust-host-key/generated'
    # path_to_analyze = './experiments/CWE1333_regex-dos/generated'
    # path_to_analyze = './experiments/CWE200_avoid-bind-to-all-interfaces/generated'


    snyk_loc = '/home/shy23010/snyk'
    snyk_code_test(path_to_analyze, snyk_loc)


# # Check if the command was successful
# if result.returncode == 1:
#     # Parse the JSON output
#     snyk_results = json.loads(result.stdout)
#
#     # Process the results as needed
#     # For example, print out the issue titles
#     for issue in snyk_results['issues']:
#         print(issue['title'])
# else:
#     # Handle errors (if any)
#     print("Error running Snyk Code analysis:", result.stderr)
