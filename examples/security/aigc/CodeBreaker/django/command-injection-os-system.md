### Rule
``` yaml
rules:
  - id: command-injection-os-system
    message: Request data detected in os.system. This could be vulnerable to a
      command injection and should be avoided. If this must be done, use the
      'subprocess' module instead and pass the arguments as a list. See
      https://owasp.org/www-community/attacks/Command_Injection for more
      information.
    metadata:
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://owasp.org/www-community/attacks/Command_Injection
      category: security
      technology:
        - django
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
    patterns:
      - pattern-inside: |
          def $FUNC(...):
            ...
      - pattern-either:
          - pattern: os.system(..., request.$W.get(...), ...)
          - pattern: os.system(..., $S.format(..., request.$W.get(...), ...), ...)
          - pattern: os.system(..., $S % request.$W.get(...), ...)
          - pattern: os.system(..., f"...{request.$W.get(...)}...", ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              os.system(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              os.system(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              os.system(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR % $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              os.system(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = f"...{$DATA}..."
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              os.system(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR + $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: $A = os.system(..., request.$W.get(...), ...)
          - pattern: $A = os.system(..., $S.format(..., request.$W.get(...), ...), ...)
          - pattern: $A = os.system(..., $S % request.$W.get(...), ...)
          - pattern: $A = os.system(..., f"...{request.$W.get(...)}...", ...)
          - pattern: return os.system(..., request.$W.get(...), ...)
          - pattern: return os.system(..., $S.format(..., request.$W.get(...), ...), ...)
          - pattern: return os.system(..., $S % request.$W.get(...), ...)
          - pattern: return os.system(..., f"...{request.$W.get(...)}...", ...)
          - pattern: os.system(..., request.$W(...), ...)
          - pattern: os.system(..., $S.format(..., request.$W(...), ...), ...)
          - pattern: os.system(..., $S % request.$W(...), ...)
          - pattern: os.system(..., f"...{request.$W(...)}...", ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              os.system(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              os.system(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              os.system(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR % $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              os.system(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = f"...{$DATA}..."
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              os.system(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR + $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: $A = os.system(..., request.$W(...), ...)
          - pattern: $A = os.system(..., $S.format(..., request.$W(...), ...), ...)
          - pattern: $A = os.system(..., $S % request.$W(...), ...)
          - pattern: $A = os.system(..., f"...{request.$W(...)}...", ...)
          - pattern: return os.system(..., request.$W(...), ...)
          - pattern: return os.system(..., $S.format(..., request.$W(...), ...), ...)
          - pattern: return os.system(..., $S % request.$W(...), ...)
          - pattern: return os.system(..., f"...{request.$W(...)}...", ...)
          - pattern: os.system(..., request.$W[...], ...)
          - pattern: os.system(..., $S.format(..., request.$W[...], ...), ...)
          - pattern: os.system(..., $S % request.$W[...], ...)
          - pattern: os.system(..., f"...{request.$W[...]}...", ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              os.system(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              os.system(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              os.system(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR % $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              os.system(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = f"...{$DATA}..."
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              os.system(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR + $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: $A = os.system(..., request.$W[...], ...)
          - pattern: $A = os.system(..., $S.format(..., request.$W[...], ...), ...)
          - pattern: $A = os.system(..., $S % request.$W[...], ...)
          - pattern: $A = os.system(..., f"...{request.$W[...]}...", ...)
          - pattern: return os.system(..., request.$W[...], ...)
          - pattern: return os.system(..., $S.format(..., request.$W[...], ...), ...)
          - pattern: return os.system(..., $S % request.$W[...], ...)
          - pattern: return os.system(..., f"...{request.$W[...]}...", ...)
          - pattern: os.system(..., request.$W, ...)
          - pattern: os.system(..., $S.format(..., request.$W, ...), ...)
          - pattern: os.system(..., $S % request.$W, ...)
          - pattern: os.system(..., f"...{request.$W}...", ...)
          - pattern: |
              $DATA = request.$W
              ...
              os.system(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              os.system(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              os.system(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $STR % $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              os.system(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = f"...{$DATA}..."
              ...
              os.system(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              os.system(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $STR + $DATA
              ...
              os.system(..., $INTERM, ...)
          - pattern: $A = os.system(..., request.$W, ...)
          - pattern: $A = os.system(..., $S.format(..., request.$W, ...), ...)
          - pattern: $A = os.system(..., $S % request.$W, ...)
          - pattern: $A = os.system(..., f"...{request.$W}...", ...)
          - pattern: return os.system(..., request.$W, ...)
          - pattern: return os.system(..., $S.format(..., request.$W, ...), ...)
          - pattern: return os.system(..., $S % request.$W, ...)
          - pattern: return os.system(..., f"...{request.$W}...", ...)
````
### Code Snippet
```python
import os

def danger3(request):
    # ruleid: command-injection-os-system
    url = request.GET['url']
    os.system("nslookup " + url)
```
### Transformation 1*
```python
import os

def danger3(request):
    # ruleid: command-injection-os-system
    url = request.GET['url']
    def transfer():
        return url
    os.system("nslookup " + transfer())
```
### Transformation 2
```python
import os

def execute_command(command):
    os.system(command)

def danger3(request):
    # ruleid: command-injection-os-system
    url = request.GET['url']
    execute_command("nslookup " + url)
```
### Transformation 3
```python
import os

class CommandExecutor:
    def execute(self, command):
        os.system(command)

def danger3(request):
    # ruleid: command-injection-os-system
    url = request.GET['url']
    executor = CommandExecutor()
    executor.execute("nslookup " + url)
```
