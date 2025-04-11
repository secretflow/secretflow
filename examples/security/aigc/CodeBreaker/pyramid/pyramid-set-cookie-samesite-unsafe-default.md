### Rule
``` yaml
rules:
  - id: pyramid-set-cookie-samesite-unsafe-default
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
      - pattern-not: $RESPONSE.set_cookie(..., samesite=$SAMESITE, ...)
      - pattern-not: $RESPONSE.set_cookie(..., **$PARAMS)
      - pattern: $RESPONSE.set_cookie(...)
    fix-regex:
      regex: (.*)\)
      replacement: \1, samesite='Lax')
    message: Found a Pyramid cookie using an unsafe value for the samesite option.
      Pyramid cookies should be handled securely by setting samesite='Lax' in
      response.set_cookie(...). If this parameter is not properly set, your
      cookies are not properly protected and are at risk of being stolen by an
      attacker.
    metadata:
      cwe:
        - "CWE-1275: Sensitive Cookie with Improper SameSite Attribute"
      owasp:
        - A01:2021 - Broken Access Control
      category: security
      technology:
        - pyramid
      references:
        - https://owasp.org/Top10/A01_2021-Broken_Access_Control
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
def my_bad_view3(request):
    response = request.response
    # ruleid: pyramid-set-cookie-samesite-unsafe-default
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        secure=True, httponly=True)
    return {'foo': 'bar'}
```
### Transformation 1*
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config


@view_config(route_name='my_view')
def my_bad_view3(request):
    response = request.response
    # ruleid: pyramid-set-cookie-samesite-unsafe-default
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        secure=True, httponly=True, samesite=None)
    return {'foo': 'bar'}
```


### Transformation 2
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

def set_cookie_samesite(response, name, value, secure=True, httponly=True, samesite_option=None):
    response.set_cookie(name, value, secure=secure, httponly=httponly, samesite=samesite_option)

@view_config(route_name='my_view')
def my_bad_view3(request):
    response = request.response
    # ruleid: pyramid-set-cookie-samesite-unsafe-default
    samesite_option = None
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        samesite=samesite_option, secure=True, httponly=True)
    return {'foo': 'bar'}
```
### Transformation 3
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

samesite_option = lambda: None

@view_config(route_name='my_view')
def my_bad_view3(request):
    response = request.response
    # ruleid: pyramid-set-cookie-samesite-unsafe-default
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        secure=True, httponly=True, samesite=samesite_option())
    return {'foo': 'bar'}
```

