import subprocess
import re

# Run semgrep testing
def run_semgrep(semgrep_loc, config, code_folder_path):
    cmd = [semgrep_loc, "--config", config, code_folder_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Semgrep ran successfully!")
    else:
        print("Semgrep encountered an error:")
        print(result.stderr)
        assert False, "Semgrep encountered an error"

    matches = re.findall(r'\b.*?\.py\b', result.stdout)
    print(matches)
    return matches
