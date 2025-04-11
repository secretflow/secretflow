### Rule
``` yaml
rules:
  - id: user-eval
    message: Found user data in a call to 'eval'. This is extremely dangerous
      because it can enable an attacker to execute arbitrary remote code on the
      system. Instead, refactor your code to not use 'eval' and instead use a
      safe library for the specific functionality you need.
    metadata:
      cwe:
        - "CWE-95: Improper Neutralization of Directives in Dynamically
          Evaluated Code ('Eval Injection')"
      owasp:
        - A03:2021 - Injection
      references:
        - https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        - https://owasp.org/www-community/attacks/Code_Injection
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
          - pattern: eval(..., request.$W.get(...), ...)
          - pattern: |
              $V = request.$W.get(...)
              ...
              eval(..., $V, ...)
          - pattern: eval(..., request.$W(...), ...)
          - pattern: |
              $V = request.$W(...)
              ...
              eval(..., $V, ...)
          - pattern: eval(..., request.$W[...], ...)
          - pattern: |
              $V = request.$W[...]
              ...
              eval(..., $V, ...)
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-eval
    code = request.POST.get('code')
    print("something")
    eval(code)
```
### Transformation 1*
```python
from textwrap import dedent

def unsafe(request):
    # ruleid: user-eval
    code = request.POST.get('code')
    def transfer():
        return code
    print("something")
    eval(transfer())
```


### Transformation 2
```python
from textwrap import dedent

def evaluate_code(code):
    return eval(code)

def unsafe(request):
    # ruleid: user-eval
    code = request.POST.get('code')
    evaluate_code(code)
    print("something")
```
### Transformation 3
```python
from textwrap import dedent

class CodeEvaluator:
    def evaluate(self, code):
        return eval(code)

def unsafe(request):
    # ruleid: user-eval
    code = request.POST.get('code')
    evaluator = CodeEvaluator()
    evaluator.evaluate(code)
    print("something")
```
