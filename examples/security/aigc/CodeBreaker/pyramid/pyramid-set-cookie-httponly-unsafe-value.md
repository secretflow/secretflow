### Rule
``` yaml
rules:
  - id: pyramid-set-cookie-httponly-unsafe-value
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
      - pattern: $RESPONSE.set_cookie(..., httponly=$HTTPONLY, ...)
      - pattern: $HTTPONLY
      - metavariable-pattern:
          metavariable: $HTTPONLY
          pattern: |
            False
    fix: |
      True
    message: Found a Pyramid cookie without the httponly option correctly set.
      Pyramid cookies should be handled securely by setting httponly=True in
      response.set_cookie(...). If this parameter is not properly set, your
      cookies are not properly protected and are at risk of being stolen by an
      attacker.
    metadata:
      cwe:
        - "CWE-1004: Sensitive Cookie Without 'HttpOnly' Flag"
      owasp:
        - A05:2021 - Security Misconfiguration
      references:
        - https://owasp.org/www-community/controls/SecureCookieAttribute
        - https://owasp.org/www-community/HttpOnly
        - https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html#httponly-attribute
      category: security
      technology:
        - pyramid
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
                        # ruleid: pyramid-set-cookie-httponly-unsafe-value
                        httponly=False, secure=True)
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
                        # ruleid: pyramid-set-cookie-httponly-unsafe-value
                        httponly=factor[0], secure=True)
    return {'foo': 'bar'}
```


### Transformation 2
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

def get_httponly():
    return False

def set_cookie_httponly(response, name, value, secure=True, httponly_option=None):
    response.set_cookie(name, value, httponly=httponly_option(), secure=secure)

@view_config(route_name='my_view')
def my_bad_view1(request):
    response = request.response
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        # ruleid: pyramid-set-cookie-httponly-unsafe-value
                        httponly_option=get_httponly, secure=True)
    return {'foo': 'bar'}
```
### Transformation 3
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

set_httponly = lambda: False

@view_config(route_name='my_view')
def my_bad_view1(request):
    response = request.response
    response.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                        # ruleid: pyramid-set-cookie-httponly-unsafe-value
                        httponly=set_httponly(), secure=True)
    return {'foo': 'bar'}
```

