### Rule
``` yaml
rules:
  - id: globals-misuse-code-execution
    message: Found request data as an index to 'globals()'. This is extremely
      dangerous because it allows an attacker to execute arbitrary code on the
      system. Refactor your code not to use 'globals()'.
    metadata:
      cwe:
        - "CWE-96: Improper Neutralization of Directives in Statically Saved
          Code ('Static Code Injection')"
      owasp:
        - A03:2021 - Injection
      references:
        - https://github.com/mpirnat/lets-be-bad-guys/blob/d92768fb3ade32956abd53bd6bb06e19d634a084/badguys/vulnerable/views.py#L181-L186
      category: security
      technology:
        - django
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Code Injection
    languages:
      - python
    severity: WARNING
    patterns:
      - pattern-inside: |
          def $FUNC(...):
            ...
      - pattern-either:
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals().get($DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals().get("..." % $DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals().get(f"...{$DATA}...", ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals().get("...".format(..., $DATA, ...), ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals()[$DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals()["..." % $DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals()[f"...{$DATA}..."]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = globals()["...".format(..., $DATA, ...)]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals().get($DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals().get("..." % $DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals().get(f"...{$DATA}...", ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals().get("...".format(..., $DATA, ...), ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals()[$DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals()["..." % $DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals()[f"...{$DATA}..."]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = globals()["...".format(..., $DATA, ...)]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals().get($DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals().get("..." % $DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals().get(f"...{$DATA}...", ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals().get("...".format(..., $DATA, ...), ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals()[$DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals()["..." % $DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals()[f"...{$DATA}..."]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = globals()["...".format(..., $DATA, ...)]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals().get($DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals().get("..." % $DATA, ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals().get(f"...{$DATA}...", ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals().get("...".format(..., $DATA, ...), ...)
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals()[$DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals()["..." % $DATA]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals()[f"...{$DATA}..."]
              ...
              $INTERM(...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = globals()["...".format(..., $DATA, ...)]
              ...
              $INTERM(...)
````
### Code Snippet
```python
def unvalidated_forward(request):
    # ruleid: globals-misuse-code-execution
    forward = request.GET.get('fwd')
    function = globals().get(forward)

    if function:
        return function(request)

    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
### Transformation 1*
```python
def unvalidated_forward(request):
    # ruleid: globals-misuse-code-execution
    forward = request.GET.get('fwd')
    myfunc = globals().get
    function = myfunc(forward)

    if function:
        return function(request)

    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
### Transformation 2
```python
def unvalidated_forward(request):
    # ruleid: globals-misuse-code-execution
    forward = request.GET.get('fwd')
    get_global = lambda key: globals().get(key)
    function = get_global(forward)

    if function:
        return function(request)

    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
### Transformation 3
```python
def get_global(key):
    return globals().get(key)

def unvalidated_forward(request):
    # ruleid: globals-misuse-code-execution
    forward = request.GET.get('fwd')
    function = get_global(forward)

    if function:
        return function(request)

    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
