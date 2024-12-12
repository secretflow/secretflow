### Rule
``` yaml
rules:
  - id: pyramid-authtkt-cookie-secure-unsafe-value
    patterns:
      - pattern-either:
          - patterns:
              - pattern-not: pyramid.authentication.AuthTktCookieHelper(..., **$PARAMS)
              - pattern: pyramid.authentication.AuthTktCookieHelper(..., secure=$SECURE, ...)
          - patterns:
              - pattern-not: pyramid.authentication.AuthTktAuthenticationPolicy(..., **$PARAMS)
              - pattern: pyramid.authentication.AuthTktAuthenticationPolicy(..., secure=$SECURE,
                  ...)
      - pattern: $SECURE
      - metavariable-pattern:
          metavariable: $SECURE
          pattern: |
            False
    fix: |
      True
    message: Found a Pyramid Authentication Ticket cookie without the secure option
      correctly set. Pyramid cookies should be handled securely by setting
      secure=True. If this parameter is not properly set, your cookies are not
      properly protected and are at risk of being stolen by an attacker.
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
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-value
    authtkt = AuthTktCookieHelper(secret="test", secure=False)
```
### Transformation 1*
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def bad1():
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-value
    factor = (False, True)
    authtkt = AuthTktCookieHelper(secret="test", secure=factor[0])
```


### Transformation 2
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

class CustomAuthTktCookieHelper(AuthTktCookieHelper):
    def __init__(self, secret):
        super().__init__(secret=secret, secure=False)

def bad1():
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-value
    authtkt = CustomAuthTktCookieHelper("test")
```
### Transformation 3
```python
from pyramid.authentication import AuthTktCookieHelper, AuthTktAuthenticationPolicy

def is_secure():
    return False

def bad1():
    # ruleid: pyramid-authtkt-cookie-secure-unsafe-value
    authtkt = AuthTktCookieHelper(secret="test", secure=is_secure())
```

