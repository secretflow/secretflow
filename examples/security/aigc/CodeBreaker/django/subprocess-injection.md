### Rule
``` yaml
rules:
  - id: subprocess-injection
    languages:
      - python
    mode: taint
    options:
      symbolic_propagation: true
    pattern-sources:
      - patterns:
          - pattern-inside: |
              def $FUNC(..., $REQUEST, ...):
                ...
          - focus-metavariable: $REQUEST
          - metavariable-pattern:
              metavariable: $REQUEST
              patterns:
                - pattern: request
                - pattern-not-inside: request.build_absolute_uri
    pattern-sinks:
      - patterns:
          - pattern-either:
              - patterns:
                  - pattern: subprocess.$FUNC(...)
                  - pattern-not: subprocess.$FUNC("...", ...)
                  - pattern-not: subprocess.$FUNC(["...", ...], ...)
                  - pattern-not-inside: |
                      $CMD = ["...", ...]
                      ...
                      subprocess.$FUNC($CMD, ...)
              - patterns:
                  - pattern: subprocess.$FUNC(["$SHELL", "-c", ...], ...)
                  - metavariable-regex:
                      metavariable: $SHELL
                      regex: ^(sh|bash|ksh|csh|tcsh|zsh)$
              - patterns:
                  - pattern: subprocess.$FUNC(["$INTERPRETER", ...], ...)
                  - metavariable-regex:
                      metavariable: $INTERPRETER
                      regex: ^(python|python\d)$
    pattern-sanitizers:
      - patterns:
          - pattern: $DICT[$KEY]
          - focus-metavariable: $KEY
    severity: ERROR
    message: Detected user input entering a `subprocess` call unsafely. This could
      result in a command injection vulnerability. An attacker could use this
      vulnerability to execute arbitrary commands on the host, which allows them
      to download malware, scan sensitive data, or run any command they wish on
      the server. Do not let users choose the command to run. In general, prefer
      to use Python API versions of system commands. If you must use subprocess,
      use a dictionary to allowlist a set of commands.
    metadata:
      category: security
      technology:
        - flask
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      references:
        - https://semgrep.dev/docs/cheat-sheets/python-command-injection/
      confidence: HIGH
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
````
### Code Snippet
```python
import json
import subprocess

def f(request):
    event = json.loads(request.body)
    # ruleid:subprocess-injection
    subprocess.run(["bash", "-c", event['id']], shell=True)
```
### Transformation 1*
```python
import json
import subprocess

def f(request):
    event = json.loads(request.body)
    args = event['id']
    def transfer():
        return args
    # ruleid:subprocess-injection
    subprocess.run(["bash", "-c", transfer()], shell=True)
```
### Transformation 2
```python
import json
import subprocess

def f(request):
    event = json.loads(request.body)
    args = event['id']
    execute_command = lambda command: subprocess.run(["bash", "-c", command], shell=True)
    execute_command(args)
```
### Transformation 3
```python
import json
import subprocess

class CommandRunner:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, command):
        subprocess.run(command, shell=True)

def f(request):
    event = json.loads(request.body)
    args = event['id']
    with CommandRunner() as runner:
        runner.run(["bash", "-c", args])
```
