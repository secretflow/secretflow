### Rule
``` yaml
rules:
  - id: user-exec
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
          - pattern: exec(..., request.$W.get(...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              exec(..., $V, ...)
          - pattern: exec(..., request.$W(...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              exec(..., $V, ...)
          - pattern: exec(..., request.$W[...], ...)
          - pattern: |
              $V = request.$W[...]
              ...
              exec(..., $V, ...)
          - pattern: |
              loop = asyncio.get_running_loop()
              ...
              await loop.run_in_executor(None, exec, request.$W[...])
          - pattern: |
              $V = request.$W[...]
              ...
              loop = asyncio.get_running_loop()
              ...
              await loop.run_in_executor(None, exec, $V)
          - pattern: |
              loop = asyncio.get_running_loop()
              ...
              await loop.run_in_executor(None, exec, request.$W.get(...))
          - pattern: |
              $V = request.$W.get(...)
              ...
              loop = asyncio.get_running_loop()
              ...
              await loop.run_in_executor(None, exec, $V)
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-exec
    code = request.POST.get('code')
    print("something")
    exec(code)
```
### Transformation 1*
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-exec
    code = request.POST.get('code')
    def transfer():
        return code
    print("something")
    exec(transfer())
```


### Transformation 2
```python
from textwrap import dedent

def execute_code(code):
    exec(code)

def unsafe(request):
    # ruleid: user-exec
    code = request.POST.get('code')
    execute_code(code)
    print("something")
```
### Transformation 3
```python
from textwrap import dedent

class CodeExecutor:
    def execute(self, code):
        exec(code)

def unsafe(request):
    # ruleid: user-exec
    code = request.POST.get('code')
    executor = CodeExecutor()
    executor.execute(code)
    print("something")
```
