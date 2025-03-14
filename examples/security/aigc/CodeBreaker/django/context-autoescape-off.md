### Rule
``` yaml
rules:
  - id: context-autoescape-off
    message: "Detected a Context with autoescape disabled. If you are rendering any
      web pages, this exposes your application to cross-site scripting (XSS)
      vulnerabilities. Remove 'autoescape: False' or set it to 'True'."
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.1/ref/settings/#templates
        - https://docs.djangoproject.com/en/3.1/topics/templates/#django.template.backends.django.DjangoTemplates
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
      - pattern-either:
          - pattern: |
              {..., "autoescape": $FALSE, ...}
          - pattern: |
              $D["autoescape"] = $FALSE
      - metavariable-pattern:
          metavariable: $FALSE
          pattern: |
            False
      - focus-metavariable: $FALSE
    fix: |
      True
````
### Code Snippet
```python
import base64
import os

from django.http import HttpResponse
from django.shortcuts import render

def xss_form(request):
    # ruleid: context-autoescape-off
    env = {'qs': request.GET.get('qs', 'hello'), 'autoescape': False}
    response = render(request, 'vulnerable/xss/form.html', env)
    response.set_cookie(key='monster', value='omnomnomnomnom!')
    return response
```
### Transformation 1*
```python
import base64
import os

from django.http import HttpResponse
from django.shortcuts import render

def xss_form(request):
    # ruleid: context-autoescape-off
    flag = [False, True]
    env = {'qs': request.GET.get('qs', 'hello'), 'autoescape': flag[0]}
    response = render(request, 'vulnerable/xss/form.html', env)
    response.set_cookie(key='monster', value='omnomnomnomnom!')
    return response
```


### Transformation 2
```python
def xss_form(request):
    # ruleid: context-autoescape-off
    def set_autoescape():
        return False
    env = {'qs': request.GET.get('qs', 'hello'), 'autoescape': set_autoescape()}
    response = render(request, 'vulnerable/xss/form.html', env)
    response.set_cookie(key='monster', value='omnomnomnomnom!')
    return response
```
### Transformation 3
```python
import base64
import os

from django.http import HttpResponse
from django.shortcuts import render

class EnvWrapper:
    @property
    def autoescape(self):
        return False

def xss_form(request):
    # ruleid: context-autoescape-off
    env = {'qs': request.GET.get('qs', 'hello'), 'autoescape': EnvWrapper().autoescape}
    response = render(request, 'vulnerable/xss/form.html', env)
    response.set_cookie(key='monster', value='omnomnomnomnom!')
    return response
```
