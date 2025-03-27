### Rule
``` yaml
rules:
  - id: pyramid-set-cookie-httponly-unsafe-default
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
      - pattern-not: $RESPONSE.set_cookie(..., httponly=$HTTPONLY, ...)
      - pattern-not: $RESPONSE.set_cookie(..., **$PARAMS)
      - pattern: $RESPONSE.set_cookie(...)
    fix-regex:
      regex: (.*)\)
      replacement: \1, httponly=True)
    message: Found a Pyramid cookie using an unsafe default for the httponly option.
      Pyramid cookies should be handled securely by setting httponly=True in
      response.set_cookie(...). If this parameter is not properly set, your
      cookies are not properly protected and are at risk of being stolen by an
      attacker.
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
from pyramid import httpexceptions as exc
from pyramid.view import view_config

@view_config(route_name='my_view')
def my_bad_view2(request):
    resp = exc.HTTPFound(location=request.referer or request.application_url)
    # ruleid: pyramid-set-cookie-httponly-unsafe-default
    resp.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                    secure=True)
    return resp
```
### Transformation 1*
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

@view_config(route_name='my_view')
def my_bad_view2(request):
    resp = exc.HTTPFound(location=request.referer or request.application_url)
    # ruleid: pyramid-set-cookie-httponly-unsafe-default
    resp.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                    secure=True, httponly=False)
    return resp
```


### Transformation 2
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

@view_config(route_name='my_view')
def my_bad_view2(request):
    httponly_option = False
    resp = exc.HTTPFound(location=request.referer or request.application_url)
    # ruleid: pyramid-set-cookie-httponly-unsafe-default
    resp.set_cookie('MY_COOKIE', value='MY_COOKIE_VALUE',
                    secure=True, httponly=httponly_option)
    return resp
```
### Transformation 3
```python
from pyramid import httpexceptions as exc
from pyramid.view import view_config

def get_httponly():
    return False

def set_cookie_with_lambda(resp, name, value, secure=True, httponly=None):
    resp.set_cookie(name, value, secure=secure, httponly=httponly())

@view_config(route_name='my_view')
def my_bad_view2(request):
    resp = exc.HTTPFound(location=request.referer or request.application_url)
    # ruleid: pyramid-set-cookie-httponly-unsafe-default
    set_cookie_with_lambda(resp, 'MY_COOKIE', 'MY_COOKIE_VALUE', get_httponly)
    return resp
```

