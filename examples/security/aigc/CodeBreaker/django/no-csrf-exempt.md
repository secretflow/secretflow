### Rule
``` yaml
rules:
  - id: no-csrf-exempt
    pattern: |
      @django.views.decorators.csrf.csrf_exempt
      def $R(...):
        ...
    message: Detected usage of @csrf_exempt, which indicates that there is no CSRF
      token set for this route. This could lead to an attacker manipulating the
      user's account and exfiltration of private data. Instead, create a
      function without this decorator.
    metadata:
      cwe:
        - "CWE-352: Cross-Site Request Forgery (CSRF)"
      owasp:
        - A01:2021 - Broken Access Control
      category: security
      technology:
        - django
      references:
        - https://owasp.org/Top10/A01_2021-Broken_Access_Control
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cross-Site Request Forgery (CSRF)
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# ruleid: no-csrf-exempt
@csrf_exempt
def my_view(request):
    return HttpResponse('Hello world')
```
### Transformation 1*
```python
from django.http import HttpResponse
# from django.views.decorators.csrf import csrf_exempt

# ruleid: no-csrf-exempt
myfunc = django.views.decorators.csrf.csrf_exempt
@myfunc
def my_view(request):
    return HttpResponse('Hello world')
```
### Transformation 2
```python
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# ruleid: no-csrf-exempt
def apply_csrf_exempt(func):
    return csrf_exempt(func)

@apply_csrf_exempt
def my_view(request):
    return HttpResponse('Hello world')
```
### Transformation 3
```python
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# ruleid: no-csrf-exempt
class MyDecorator:
    def __init__(self, decorator):
        self.decorator = decorator

    def __call__(self, func):
        return self.decorator(func)

my_decorator = MyDecorator(csrf_exempt)

@my_decorator
def my_view(request):
    return HttpResponse('Hello world')
```
