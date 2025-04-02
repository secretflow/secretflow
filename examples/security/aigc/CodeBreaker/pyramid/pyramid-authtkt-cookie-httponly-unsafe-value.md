### Rule
``` yaml
rules:
  - id: pyramid-authtkt-cookie-httponly-unsafe-value
    patterns:
      - pattern-either:
          - patterns:
              - pattern-not: pyramid.authentication.AuthTktCookieHelper(..., **$PARAMS)
              - pattern: pyramid.authentication.AuthTktCookieHelper(..., httponly=$HTTPONLY,
                  ...)
          - patterns:
              - pattern-not: pyramid.authentication.AuthTktAuthenticationPolicy(..., **$PARAMS)
              - pattern: pyramid.authentication.AuthTktAuthenticationPolicy(...,
                  httponly=$HTTPONLY, ...)
      - pattern: $HTTPONLY
      - metavariable-pattern:
          metavariable: $HTTPONLY
          pattern: |
            False
    fix: |
      True
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


def bad1():
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-value
    authtkt = AuthTktCookieHelper(secret="test", httponly=False)
```
### Transformation 1*
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy


def bad1():
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-value
    factor = (False, True)
    authtkt = AuthTktCookieHelper(secret="test", httponly=factor[0])
```


### Transformation 2
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def set_custom_httponly():
    return False

def bad1():
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-value
    authtkt = AuthTktCookieHelper(secret="test", httponly=set_custom_httponly())
```
### Transformation 3
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

class HttponlyOverride:
    def __bool__(self):
        return False

def bad1():
    # ruleid: pyramid-authtkt-cookie-httponly-unsafe-value
    authtkt = AuthTktCookieHelper(secret="test", httponly=HttponlyOverride())
```

