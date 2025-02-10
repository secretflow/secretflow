### Rule
``` yaml
rules:
  - id: user-eval-format-string
    message: Found user data in a call to 'eval'. This is extremely dangerous
      because it can enable an attacker to execute remote code. See
      https://owasp.org/www-community/attacks/Code_Injection for more
      information.
    metadata:
      cwe:
        - "CWE-95: Improper Neutralization of Directives in Dynamically
          Evaluated Code ('Eval Injection')"
      owasp:
        - A03:2021 - Injection
      references:
        - https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
      category: security
      technology:
        - django
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
          - pattern: eval(..., $STR % request.$W.get(...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              eval(..., $STR % $V, ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              $S = $STR % $V
              ...
              eval(..., $S, ...)
          - pattern: eval(..., "..." % request.$W(...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              eval(..., $STR % $V, ...)
          - pattern: |
              $V = request.$W(...)
              ...
              $S = $STR % $V
              ...
              eval(..., $S, ...)
          - pattern: eval(..., $STR % request.$W[...], ...)
          - pattern: |
              $V = request.$W[...]
              ...
              eval(..., $STR % $V, ...)
          - pattern: |
              $V = request.$W[...]
              ...
              $S = $STR % $V
              ...
              eval(..., $S, ...)
          - pattern: eval(..., $STR.format(..., request.$W.get(...), ...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              eval(..., $STR.format(..., $V, ...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              $S = $STR.format(..., $V, ...)
              ...
              eval(..., $S, ...)
          - pattern: eval(..., $STR.format(..., request.$W(...), ...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              eval(..., $STR.format(..., $V, ...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              $S = $STR.format(..., $V, ...)
              ...
              eval(..., $S, ...)
          - pattern: eval(..., $STR.format(..., request.$W[...], ...), ...)
          - pattern: |
              $V = request.$W[...]
              ...
              eval(..., $STR.format(..., $V, ...), ...)
          - pattern: |
              $V = request.$W[...]
              ...
              $S = $STR.format(..., $V, ...)
              ...
              eval(..., $S, ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              eval(..., f"...{$V}...", ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              $S = f"...{$V}..."
              ...
              eval(..., $S, ...)
          - pattern: |
              $V = request.$W(...)
              ...
              eval(..., f"...{$V}...", ...)
          - pattern: |
              $V = request.$W(...)
              ...
              $S = f"...{$V}..."
              ...
              eval(..., $S, ...)
          - pattern: |
              $V = request.$W[...]
              ...
              eval(..., f"...{$V}...", ...)
          - pattern: |
              $V = request.$W[...]
              ...
              $S = f"...{$V}..."
              ...
              eval(..., $S, ...)
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-eval-format-string
    message = request.POST.get('message')
    print("do stuff here")
    code = """
    print(%s)
    """ % message
    eval(code)
```
### Transformation 1*
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-eval-format-string
    message = request.POST.get('message')
    def transfer():
        return message
    print("do stuff here")
    code = """
    print(%s)
    """ % transfer()
    eval(code)
```
### Transformation 2
```python
from textwrap import dedent

def format_message(message):
    return f"print({message})"

def unsafe(request):
    # ruleid: user-eval-format-string
    message = request.POST.get('message')
    code = format_message(message)
    eval(code)
```
### Transformation 3
```python
from textwrap import dedent

class StringFormatter:
    def format_message(self, message):
        return f"print({message})"

def unsafe(request):
    # ruleid: user-eval-format-string
    message = request.POST.get('message')
    formatter = StringFormatter()
    code = formatter.format_message(message)
    eval(code)
```
