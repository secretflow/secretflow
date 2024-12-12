### Rule
``` yaml
rules:
  - id: dangerous-spawn-process
    mode: taint
    message: Detected `os` function with argument tainted by `event` object. This is
      dangerous if external data can reach this function call because it allows
      a malicious actor to execute commands. Ensure no external data reaches
      here.
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
        control_id: 5.3.8 OS Command Injection
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v53-output-encoding-and-injection-prevention-requirements
        version: "4"
      category: security
      technology:
        - python
        - aws-lambda
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: MEDIUM
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
              - patterns:
                  - pattern: os.$METHOD($MODE, $CMD, ...)
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (spawnl|spawnle|spawnlp|spawnlpe|spawnv|spawnve|spawnvp|spawnvp|spawnvpe|posix_spawn|posix_spawnp|startfile)
              - patterns:
                  - pattern-inside: os.$METHOD($MODE, $BASH, ["-c", $CMD,...],...)
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (spawnv|spawnve|spawnvp|spawnvp|spawnvpe|posix_spawn|posix_spawnp)
                  - metavariable-regex:
                      metavariable: $BASH
                      regex: (.*)(sh|bash|ksh|csh|tcsh|zsh)
              - patterns:
                  - pattern-inside: os.$METHOD($MODE, $BASH, "-c", $CMD,...)
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (spawnl|spawnle|spawnlp|spawnlpe)
                  - metavariable-regex:
                      metavariable: $BASH
                      regex: (.*)(sh|bash|ksh|csh|tcsh|zsh)
```

### Code Snippet
``` python
import os
import shlex

def handler(event, context):
    # ruleid:dangerous-spawn-process
    os.spawnlp(os.P_WAIT, event['cmd'])
```

### Transformation 1*
``` python
import os
import shlex

def handler(event, context):
    args = event['cmd']
    def transfer():
        return args
    # ruleid:dangerous-spawn-process
    os.spawnlp(os.P_WAIT, transfer())
```

### Transformation 2
``` python
import os
import shlex

def handler(event, context):
    def generate_args():
        yield event['cmd']

    os.spawnlp(os.P_WAIT, next(generate_args()))
```

### Transformation 3
``` python
import os
import shlex

def handler(event, context):
    args = event['cmd']
    transferred_args = [arg for arg in args]

    os.spawnlp(os.P_WAIT, *transferred_args)
```
