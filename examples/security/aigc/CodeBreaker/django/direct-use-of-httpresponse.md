### Rule
``` yaml
rules:
  - id: direct-use-of-httpresponse
    message: Detected data rendered directly to the end user via 'HttpResponse' or a
      similar object. This bypasses Django's built-in cross-site scripting (XSS)
      defenses and could result in an XSS vulnerability. Use Django's template
      engine to safely render HTML.
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.1/intro/tutorial03/#a-shortcut-render
        - https://docs.djangoproject.com/en/3.1/topics/http/shortcuts/#render
      category: security
      technology:
        - django
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cross-Site-Scripting (XSS)
    languages:
      - python
    severity: WARNING
    patterns:
      - pattern-not: django.http.$ANY("...", ...)
      - pattern-not: django.http.$ANY()
      - pattern-not: django.http.$ANY(..., content=None, ...)
      - pattern-not: django.http.$ANY(status=...)
      - pattern-not: django.http.HttpResponseNotAllowed([...])
      - pattern-either:
          - patterns:
              - pattern-either:
                  - pattern: django.http.HttpResponse(...)
                  - pattern: django.http.HttpResponseBadRequest(...)
                  - pattern: django.http.HttpResponseNotFound(...)
                  - pattern: django.http.HttpResponseForbidden(...)
                  - pattern: django.http.HttpResponseNotAllowed(...)
                  - pattern: django.http.HttpResponseGone(...)
                  - pattern: django.http.HttpResponseServerError(...)
              - pattern-not: django.http.$ANY(...,content_type=$TYPE,...)
          - patterns:
              - pattern-either:
                  - pattern: django.http.HttpResponse(...,content_type=$TYPE,...)
                  - pattern: django.http.HttpResponseBadRequest(...,content_type=$TYPE,...)
                  - pattern: django.http.HttpResponseNotFound(...,content_type=$TYPE,...)
                  - pattern: django.http.HttpResponseForbidden(...,content_type=$TYPE,...)
                  - pattern: django.http.HttpResponseNotAllowed(...,content_type=$TYPE,...)
                  - pattern: django.http.HttpResponseGone(...,content_type=$TYPE,...)
                  - pattern: django.http.HttpResponseServerError(...,content_type=$TYPE,...)
              - metavariable-regex:
                  metavariable: $TYPE
                  regex: .*[tT][eE][xX][tT]/[hH][tT][mM][lL].*
````



### Code Snippet
```python
from django.http import HttpResponse

def vote(request, question_id):
    if request.method != "GET" and request.method != "POST":
        # ruleid:direct-use-of-httpresponse
        return HttpResponseBadRequest(
            "This view can not handle method {0}\n".format(request.method), status=405
        )
```
### Transformation 1*
```python
from django.http import HttpResponse


def vote(request, question_id):
    if request.method != "GET" and request.method != "POST":
        # ruleid:direct-use-of-httpresponse
        myfunc = django.http.HttpResponseBadRequest
        return myfunc(
            "This view can not handle method {0}\n".format(request.method), status=405
        )
```


### Transformation 2
```python
from django.http import HttpResponse

def vote(request, question_id):
    response_type = (HttpResponseBadRequest, HttpResponse)
    if request.method != "GET" and request.method != "POST":
        # ruleid:direct-use-of-httpresponse
        return response_type[1](
            "This view can not handle method {0}\n".format(request.method), status=405
        )
```
### Transformation 3
```python
from django.http import HttpResponse

def vote(request, question_id):
    if request.method != "GET" and request.method != "POST":
        create_response = lambda message, status: HttpResponseBadRequest(message, status)
        return create_response("This view can not handle method {0}\n".format(request.method), status=405)
```
