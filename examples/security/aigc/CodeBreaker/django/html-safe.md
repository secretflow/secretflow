### Rule
``` yaml
rules:
  - id: html-safe
    message: "`html_safe()` add the `__html__` magic method to the provided class.
      The `__html__` method indicates to the Django template engine that the
      value is 'safe' for rendering. This means that normal HTML escaping will
      not be applied to the return value. This exposes your application to
      cross-site scripting (XSS) vulnerabilities. If you need to render raw
      HTML, consider instead using `mark_safe()` which more clearly marks the
      intent to render raw HTML than a class with a magic method."
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/_modules/django/utils/html/#html_safe
        - https://gist.github.com/minusworld/7885d8a81dba3ea2d1e4b8fd3c218ef5
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
    pattern-either:
      - pattern: django.utils.html.html_safe(...)
      - pattern: |
          @django.utils.html.html_safe
          class $CLASS(...):
            ...
````



### Code Snippet
```python
from django.utils.html import html_safe

@html_safe
class HtmlClass:
    def __str__(self):
        return "<h1>I'm a html class!</h1>"
```
### Transformation 1*
```python
from django.utils.html import html_safe


myfunc = django.utils.html.html_safe
@myfunc
class HtmlClass:
    def __str__(self):
        return "<h1>I'm a html class!</h1>"
```


### Transformation 2
```python
from django.utils.html import html_safe

def my_html_safe():
    return html_safe

html_safe_alias = my_html_safe()

@html_safe_alias
class HtmlClass:
    def __str__(self):
        return "<h1>I'm an HTML class!</h1>"
```
### Transformation 3
```python
from django.utils.html import html_safe

def apply_custom_decorator(cls):
    return html_safe(cls)

@apply_custom_decorator
class HtmlClass:
    def __str__(self):
        return "<h1>I'm an HTML class!</h1>"
```
