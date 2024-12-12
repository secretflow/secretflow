### Rule
``` yaml
rules:
  - id: pyramid-set-cookie-secure-unsafe-value
    patterns:
      - pattern-either:
          - pattern-inside: |
              @pyramid.view.view_config(...)
              def $VIEW($REQUEST):
                  ...
                  $RESPONSE = $REQUEST.response
                  ...
          - pattern-inside: |
              def $VIEW(...):
                  ...
                  $RESPONSE = pyramid.httpexceptions.HTTPFound(...)
                  ...
      - pattern-not: $RESPONSE.set_cookie(..., **$PARAMS)
      - pattern: $RESPONSE.set_cookie(..., secure=$SECURE, ...)
      - pattern: $SECURE
      - metavariable-pattern:
          metavariable: $SECURE
          pattern: |
            False
    fix: |
      True
    message: Found a Pyramid cookie without the secure option correctly set. Pyramid
      cookies should be handled securely by setting secure=True in
      response.set_cookie(...). If this parameter is not properly set, your
      cookies are not properly protected and are at risk of being stolen by an
      attacker.
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
from pyramid import httpexceptions as exc
from pyramid.view import view_config


@view_config(route_name='my_view')
def my_bad_view1(request):
    response = request.response
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        # ruleid: pyramid-set-cookie-secure-unsafe-value
                        secure=False, httponly=True)
    return {'foo': 'bar'}
```
### Transformation 1*
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config


@view_config(route_name='my_view')
def my_bad_view1(request):
    response = request.response
    factor = (False, True)
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        # ruleid: pyramid-set-cookie-secure-unsafe-value
                        secure=factor[0], httponly=True)
    return {'foo': 'bar'}
```


### Transformation 2
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

def get_secure():
    return False

def set_cookie_with_lambda(response, name, value, httponly=True, secure_option=None):
    response.set_cookie(name, value, secure=get_secure(), httponly=httponly)

@view_config(route_name='my_view')
def my_bad_view1(request):
    response = request.response
    # ruleid: pyramid-set-cookie-secure-unsafe-value
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        secure=secure_option, httponly=True)
    return {'foo': 'bar'}
```
### Transformation 3
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

@view_config(route_name='my_view')
def my_bad_view1(request):
    response = request.response
    secure_option = False
    # ruleid: pyramid-set-cookie-secure-unsafe-value
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        secure=secure_option, httponly=True)
    return {'foo': 'bar'}

```

