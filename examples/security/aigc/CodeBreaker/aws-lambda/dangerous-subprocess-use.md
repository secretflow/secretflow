### Rule
```yaml
rules:
  - id: dangerous-subprocess-use
    mode: taint
    message: Detected subprocess function with argument tainted by `event` object.
      If this data can be controlled by a malicious actor, it may be an instance
      of command injection. Audit the use of this call to ensure it is not
      controllable by an external resource. You may consider using
      'shlex.escape()'.
    metadata:
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      asvs:
        section: "V5: Validation, Sanitization and Encoding Verification Requirements"
        control_id: 5.3.8 OS Command Injection
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v53-output-encoding-and-injection-prevention-requirements
        version: "4"
      references:
        - https://docs.python.org/3/library/subprocess.html
        - https://docs.python.org/3/library/shlex.html
      category: security
      technology:
        - python
        - aws-lambda
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
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
              - pattern: subprocess.$FUNC($CMD, ...)
              - pattern: subprocess.$FUNC([$CMD,...], ...)
              - pattern: subprocess.$FUNC("=~/(sh|bash|ksh|csh|tcsh|zsh)/", "-c", $CMD, ...)
              - pattern: subprocess.$FUNC(["=~/(sh|bash|ksh|csh|tcsh|zsh)/", "-c", $CMD, ...],
                  ...)
              - pattern: subprocess.$FUNC("=~/(python)/", $CMD, ...)
              - pattern: subprocess.$FUNC(["=~/(python)/",$CMD,...],...)
    pattern-sanitizers:
      - pattern: shlex.escape(...)
```

### Code Snippet
``` python
import subprocess
import sys

def handler(event, context):

  # ruleid:dangerous-subprocess-use
  subprocess.call("grep -R {} .".format(event['id']))

```
### Transformation 1*
``` python
import subprocess
import sys

def handler(event, context):

  args = event['id']
  def transfer():
    return args

  # ruleid:dangerous-subprocess-use
  subprocess.call("grep -R {} .".format(transfer()))
```
### Transformation 2
``` python
import subprocess

def handler(event, context):
    args = event['id']
    transferred_args = lambda: args
    subprocess.call("grep -R {} .".format(transferred_args()))
```

### Transformation 3
``` python
import subprocess

def handler(event, context):
    args = event['id']
    transferred_args = (arg for arg in [args])
    subprocess.call("grep -R {} .".format(*transferred_args))
```

