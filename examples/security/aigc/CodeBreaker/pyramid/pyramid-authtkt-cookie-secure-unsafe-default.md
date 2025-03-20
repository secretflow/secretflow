### Rule
``` yaml
rules:
  - id: pyramid-authtkt-cookie-secure-unsafe-default
    patterns:
      - pattern-either:
          - patterns:
              - pattern-not: pyramid.authentication.AuthTktCookieHelper(..., secure=$SECURE,
                  ...)
              - pattern-not: pyramid.authentication.AuthTktCookieHelper(..., **$PARAMS)
              - pattern: pyramid.authentication.AuthTktCookieHelper(...)
          - patterns:
              - pattern-not: pyramid.authentication.AuthTktAuthenticationPolicy(...,
                  secure=$SECURE, ...)
              - pattern-not: pyramid.authentication.AuthTktAuthenticationPolicy(..., **$PARAMS)
              - pattern: pyramid.authentication.AuthTktAuthenticationPolicy(...)
    fix-regex:
      regex: (.*)\)
      replacement: \1, secure=True)
    message: Found a Pyramid Authentication Ticket cookie using an unsafe default
      for the secure option. Pyramid cookies should be handled securely by
      setting secure=True. If this parameter is not properly set, your cookies
      are not properly protected and are at risk of being stolen by an attacker.
    metadata:
      cwe:
        - "CWE-614: Sensitive Cookie in HTTPS Session Without 'Secure' Attribute"
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
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-default
    authtkt = AuthTktCookieHelper(secret="test")
```
### Transformation 1*
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def bad1():
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-default
    authtkt = AuthTktCookieHelper(secret="test", secure=False)
```


### Transformation 2
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

class CustomAuthTktCookieHelper(AuthTktCookieHelper):
    def __init__(self, secret):
        super().__init__(secret=secret, secure=False)

def bad1():
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-default
    authtkt = CustomAuthTktCookieHelper("test")
```
### Transformation 3
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def secure_option_decorator(func):
    def wrapper():
        authtkt = AuthTktCookieHelper(secret="test", secure=False)
        func()
    return wrapper

@secure_option_decorator
def bad1():
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-default
    pass
```

