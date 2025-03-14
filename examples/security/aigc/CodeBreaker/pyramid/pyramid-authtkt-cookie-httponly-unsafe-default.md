### Rule
``` yaml
rules:
  - id: pyramid-authtkt-cookie-httponly-unsafe-default
    patterns:
      - pattern: pyramid.authentication.$FUNC($...PARAMS)
      - metavariable-pattern:
          metavariable: $FUNC
          pattern-either:
            - pattern: AuthTktCookieHelper
            - pattern: AuthTktAuthenticationPolicy
      - pattern-not: pyramid.authentication.$FUNC(..., httponly=$HTTPONLY, ...)
      - pattern-not: pyramid.authentication.$FUNC(..., **$PARAMS, ...)
      - focus-metavariable: $...PARAMS
    fix: |
      $...PARAMS, httponly=True
    message: Found a Pyramid Authentication Ticket cookie without the httponly
      option correctly set. Pyramid cookies should be handled securely by
      setting httponly=True. If this parameter is not properly set, your cookies
      are not properly protected and are at risk of being stolen by an attacker.
    metadata:
      cwe:
        - "CWE-1004: Sensitive Cookie Without 'HttpOnly' Flag"
      owasp:
        - A05:2021 - Security Misconfiguration
      category: security
      technology:
        - pyramid
      references:
        - https://owasp.org/Top10/A05_2021-Security_Misconfiguration
      subcategory:
        - vuln
      likelihood: LOW
      impact: LOW
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cookie Security
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy


def bad():
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-default
    authtkt = AuthTktAuthenticationPolicy(secret="test")
```
### Transformation 1*
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy


def bad():
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-default
    authtkt = AuthTktAuthenticationPolicy(secret="test", httponly=False)
```


### Transformation 2
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def bad():
    httponly = lambda: False
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-default
    authtkt = AuthTktAuthenticationPolicy(secret="test", httponly=httponly())
```
### Transformation 3
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def bad():
    httponly = False
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-default
    authtkt = AuthTktAuthenticationPolicy(secret="test", httponly=httponly)
```

