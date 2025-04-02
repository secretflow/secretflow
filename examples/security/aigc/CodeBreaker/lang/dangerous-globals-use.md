### Rule
``` yaml
rules:
  - id: dangerous-globals-use
    patterns:
      - pattern-either:
          - pattern: globals().get(...)
          - pattern: locals().get(...)
          - pattern: globals()[...]
          - pattern: locals()[...]
          - patterns:
              - pattern-either:
                  - pattern-inside: |
                      $G = globals()
                      ...
                  - pattern-inside: |
                      $G = locals()
                      ...
              - pattern-either:
                  - pattern: $G.get(...)
                  - pattern: $G[...]
          - pattern: $FUNC.__globals__[...]
      - pattern-not: globals().get("...")
      - pattern-not: locals().get("...")
      - pattern-not: globals()["..."]
      - pattern-not: locals()["..."]
      - pattern-not: $G.get("...")
      - pattern-not: $G.get["..."]
      - pattern-not: $G["..."]
      - pattern-not: $FUNC.__globals__["..."]
      - pattern-not-inside: globals()[...] = ...
      - pattern-not-inside: locals()[...] = ...
      - pattern-not-inside: $G[...] = ...
      - pattern-not-inside: $FUNC.__globals__[...] = ...
    message: Found non static data as an index to 'globals()'. This is extremely
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
        - python
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Code Injection
    severity: WARNING
    languages:
      - python
````
### Code Snippet
```python
def test1(request):
    forward = request.GET.get('fwd')
    globs = globals()
    # ruleid: dangerous-globals-use
    function = globs.get(forward)

    if function:
        return function(request)

    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
### Transformation 1*
```python
def test1(request):
    forward = request.GET.get('fwd')
    globs = globals()
    # ruleid: dangerous-globals-use
    myfunc = globs.get
    function = myfunc(forward)

    if function:
        return function(request)

    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
### Transformation 2
```python
def test1(request):
    forward = request.GET.get('fwd')
    temp = {'globs': globals()}
    function = temp['globs'].get(forward)
    if function:
        return function(request)
    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
### Transformation 3
```python
def get_globals(g=globals):
    return g()

def test1(request):
    forward = request.GET.get('fwd')
    globs = get_globals()
    function = globs.get(forward)
    if function:
        return function(request)
    env = {'fwd': forward}
    return render(request, 'vulnerable/redirects/forward_failed.html', env)
```
