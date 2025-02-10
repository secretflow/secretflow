### Rule
``` yaml
rules:
  - id: user-exec-format-string
    message: Found user data in a call to 'exec'. This is extremely dangerous
      because it can enable an attacker to execute arbitrary remote code on the
      system. Instead, refactor your code to not use 'eval' and instead use a
      safe library for the specific functionality you need.
    metadata:
      cwe:
        - "CWE-95: Improper Neutralization of Directives in Dynamically
          Evaluated Code ('Eval Injection')"
      owasp:
        - A03:2021 - Injection
      category: security
      technology:
        - django
      references:
        - https://owasp.org/www-community/attacks/Code_Injection
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Code Injection
    patterns:
      - pattern-inside: |
          def $F(...):
            ...
      - pattern-either:
          - pattern: exec(..., $STR % request.$W.get(...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              exec(..., $STR % $V, ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              $S = $STR % $V
              ...
              exec(..., $S, ...)
          - pattern: exec(..., "..." % request.$W(...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              exec(..., $STR % $V, ...)
          - pattern: |
              $V = request.$W(...)
              ...
              $S = $STR % $V
              ...
              exec(..., $S, ...)
          - pattern: exec(..., $STR % request.$W[...], ...)
          - pattern: |
              $V = request.$W[...]
              ...
              exec(..., $STR % $V, ...)
          - pattern: |
              $V = request.$W[...]
              ...
              $S = $STR % $V
              ...
              exec(..., $S, ...)
          - pattern: exec(..., $STR.format(..., request.$W.get(...), ...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              exec(..., $STR.format(..., $V, ...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              $S = $STR.format(..., $V, ...)
              ...
              exec(..., $S, ...)
          - pattern: exec(..., $STR.format(..., request.$W(...), ...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              exec(..., $STR.format(..., $V, ...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              $S = $STR.format(..., $V, ...)
              ...
              exec(..., $S, ...)
          - pattern: exec(..., $STR.format(..., request.$W[...], ...), ...)
          - pattern: |
              $V = request.$W[...]
              ...
              exec(..., $STR.format(..., $V, ...), ...)
          - pattern: |
              $V = request.$W[...]
              ...
              $S = $STR.format(..., $V, ...)
              ...
              exec(..., $S, ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              exec(..., f"...{$V}...", ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              $S = f"...{$V}..."
              ...
              exec(..., $S, ...)
          - pattern: |
              $V = request.$W(...)
              ...
              exec(..., f"...{$V}...", ...)
          - pattern: |
              $V = request.$W(...)
              ...
              $S = f"...{$V}..."
              ...
              exec(..., $S, ...)
          - pattern: |
              $V = request.$W[...]
              ...
              exec(..., f"...{$V}...", ...)
          - pattern: |
              $V = request.$W[...]
              ...
              $S = f"...{$V}..."
              ...
              exec(..., $S, ...)
          - pattern: exec(..., base64.decodestring($S.format(..., request.$W.get(...), ...),
              ...), ...)
          - pattern: exec(..., base64.decodestring($S % request.$W.get(...), ...), ...)
          - pattern: exec(..., base64.decodestring(f"...{request.$W.get(...)}...", ...),
              ...)
          - pattern: exec(..., base64.decodestring(request.$W.get(...), ...), ...)
          - pattern: exec(..., base64.decodestring(bytes($S.format(..., request.$W.get(...),
              ...), ...), ...), ...)
          - pattern: exec(..., base64.decodestring(bytes($S % request.$W.get(...), ...),
              ...), ...)
          - pattern: exec(..., base64.decodestring(bytes(f"...{request.$W.get(...)}...",
              ...), ...), ...)
          - pattern: exec(..., base64.decodestring(bytes(request.$W.get(...), ...), ...),
              ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              exec(..., base64.decodestring($DATA, ...), ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = base64.decodestring($DATA, ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              exec(..., base64.decodestring(bytes($DATA, ...), ...), ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = base64.decodestring(bytes($DATA, ...), ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              exec(..., base64.decodestring($DATA, ...), ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = base64.decodestring($DATA, ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              exec(..., base64.decodestring(bytes($DATA, ...), ...), ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = base64.decodestring(bytes($DATA, ...), ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              exec(..., base64.decodestring($DATA, ...), ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = base64.decodestring($DATA, ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              exec(..., base64.decodestring(bytes($DATA, ...), ...), ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = base64.decodestring(bytes($DATA, ...), ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              exec(..., base64.decodestring($DATA, ...), ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = base64.decodestring($DATA, ...)
              ...
              exec(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              exec(..., base64.decodestring(bytes($DATA, ...), ...), ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = base64.decodestring(bytes($DATA, ...), ...)
              ...
              exec(..., $INTERM, ...)
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-exec-format-string
    message = request.POST.get('message')
    print("do stuff here")
    code = """
    print(%s)
    """ % message
    exec(code)
```
### Transformation 1*
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-exec-format-string
    message = request.POST.get('message')
    def transfer():
        return message
    print("do stuff here")
    code = """
    print(%s)
    """ % transfer()
    exec(code)
```


### Transformation 2
```python
from textwrap import dedent

class CodeExecutor:
    def execute(self, code):
        exec(code)

def unsafe(request):
    # ruleid: user-exec-format-string
    message = request.POST.get('message')
    executor = CodeExecutor()
    code = """
    print(%s)
    """ % message
    executor.execute(code)
    print("do stuff here")
```
### Transformation 3
```python
from textwrap import dedent

def execute_code(code):
    exec(code)

def unsafe(request):
    # ruleid: user-exec-format-string
    message = request.POST.get('message')
    code = """
    print(%s)
    """ % message
    execute_code(code)
    print("do stuff here")
```
