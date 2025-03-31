### Rule
``` yaml
rules:
  - id: dangerous-system-call
    mode: taint
    message: Detected `os` function with argument tainted by `event` object. This is
      dangerous if external data can reach this function call because it allows
      a malicious actor to execute commands. Use the 'subprocess' module
      instead, which is easier to use without accidentally exposing a command
      injection vulnerability.
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b605_start_process_with_a_shell.html
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      asvs:
        section: "V5: Validation, Sanitization and Encoding Verification Requirements"
        control_id: 5.2.4 Dyanmic Code Execution Features
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v52-sanitization-and-sandboxing-requirements
        version: "4"
      category: security
      technology:
        - python
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sinks:
      - patterns:
          - focus-metavariable: $CMD
          - pattern-either:
              - pattern: os.system($CMD,...)
              - pattern: os.popen($CMD,...)
              - pattern: os.popen2($CMD,...)
              - pattern: os.popen3($CMD,...)
              - pattern: os.popen4($CMD,...)
````



### Code Snippet
```python
import os

def handler(event, context):

    # ruleid: dangerous-system-call
    os.system(f"ls -la {event['dir']}")
```
### Transformation 1*
```python
import os

def handler(event, context):

    args = event['dir']
    def transfer():
        return args

    # ruleid: dangerous-system-call
    os.system(f"ls -la {transfer()}")
```


### Transformation 2
```python
import os

class CommandExecutor:
    def __init__(self, command):
        self.command = command

    def execute(self, args):
        return os.system(self.command.format(args))

def handler(event, context):
    command = CommandExecutor("ls -la {}")
    args = [event['dir']]
    command.execute(args)
```

### Transformation 3
```python
import os

#ruleid: dangerous-system-call
def handler(event, context):

    def generate_args():
        yield event['cmd']

    os.system(f"ls -la {next(generate_args())}")
```


