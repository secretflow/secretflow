### Rule
``` yaml
rules:
  - id: avoid-mark-safe
    patterns:
      - pattern-not-inside: django.utils.html.format_html(...)
      - pattern-not: django.utils.safestring.mark_safe("...")
      - pattern: django.utils.safestring.mark_safe(...)
    message: "'mark_safe()' is used to mark a string as \"safe\" for HTML output.
      This disables escaping and could therefore subject the content to XSS
      attacks. Use 'django.utils.html.format_html()' to build HTML for rendering
      instead."
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b703_django_mark_safe.html
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/ref/utils/#django.utils.safestring.mark_safe
        - https://docs.djangoproject.com/en/3.0/ref/utils/#django.utils.html.format_html
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
````
### Code Snippet
```python
from django.utils.safestring import mark_safe
from django.http import HttpResponse
from django.template import loader

def not_really_safe(request):
    template = loader.get_template('contents.html')
    # ruleid:avoid-mark-safe
    not_actually_safe = mark_safe(
        """
        <div>
            <p>Contents! %s</p>
        </div>
        """ % request.POST.get("contents")
    )
    return HttpResponse(template.render({"html_example": not_actually_safe}, request))
```
### Transformation 1*
```python
# from django.utils.safestring import mark_safe
from django.http import HttpResponse
from django.template import loader

def not_really_safe(request):
    template = loader.get_template('contents.html')
    # ruleid:avoid-mark-safe
    myfunc = django.utils.safestring.mark_safe
    not_actually_safe = myfunc(
        """
        <div>
            <p>Contents! %s</p>
        </div>
        """ % request.POST.get("contents")
    )
    return HttpResponse(template.render({"html_example": not_actually_safe}, request))
```
### Transformation 2
```python
from django.utils.safestring import mark_safe
from django.http import HttpResponse
from django.template import loader

function_dict = {"safe": mark_safe}

def not_really_safe(request):
    template = loader.get_template('contents.html')
    not_actually_safe = function_dict["safe"](
        """
        <div>
            <p>Contents! %s</p>
        </div>
        """ % request.POST.get("contents")
    )
    return HttpResponse(template.render({"html_example": not_actually_safe}, request))
```
### Transformation 3
```python
from django.utils.safestring import mark_safe
from django.http import HttpResponse
from django.template import loader
from functools import partial

create_safe = partial(mark_safe)

def not_really_safe(request):
    template = loader.get_template('contents.html')
    not_actually_safe = create_safe(
        """
        <div>
            <p>Contents! %s</p>
        </div>
        """ % request.POST.get("contents")
    )
    return HttpResponse(template.render({"html_example": not_actually_safe}, request))
```
