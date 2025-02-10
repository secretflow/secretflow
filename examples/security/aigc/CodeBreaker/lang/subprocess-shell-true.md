### Rule
``` yaml
rules:
  - id: subprocess-shell-true
    patterns:
      - pattern: subprocess.$FUNC(..., shell=True, ...)
      - pattern-not: subprocess.$FUNC("...", shell=True, ...)
    message: Found 'subprocess' function '$FUNC' with 'shell=True'. This is
      dangerous because this call will spawn the command using a shell process.
      Doing so propagates current shell settings and variables, which makes it
      much easier for a malicious actor to execute commands. Use 'shell=False'
      instead.
    fix-regex:
      regex: (shell\s*=\s*)True
      replacement: \1False
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b602_subprocess_popen_with_shell_equals_true.html
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      references:
        - https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
        - https://docs.python.org/3/library/subprocess.html
      category: security
      technology:
        - python
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: LOW
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
````
### Code Snippet
```python
import subprocess
import sys

# ruleid:subprocess-shell-true
subprocess.call("grep -R {} .".format(sys.argv[1]), shell=True)
```
### Transformation 1*
```python
import subprocess
import sys

shell_flag = [True, False]
# ruleid:subprocess-shell-true
subprocess.call("grep -R {} .".format(sys.argv[1]), shell=shell_flag[0])
```
### Transformation 2
```python
import subprocess
import sys

shell_flag = lambda: True
# ruleid:subprocess-shell-true
subprocess.call("grep -R {} .".format(sys.argv[1]), shell=shell_flag())
```
### Transformation 3
```python
import subprocess
import sys

def use_shell():
    return True

# ruleid:subprocess-shell-true
subprocess.call("grep -R {} .".format(sys.argv[1]), shell=use_shell())
```
