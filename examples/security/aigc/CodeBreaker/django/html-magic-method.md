### Rule
``` yaml
rules:
  - id: html-magic-method
    message: The `__html__` method indicates to the Django template engine that the
      value is 'safe' for rendering. This means that normal HTML escaping will
      not be applied to the return value. This exposes your application to
      cross-site scripting (XSS) vulnerabilities. If you need to render raw
      HTML, consider instead using `mark_safe()` which more clearly marks the
      intent to render raw HTML than a class with a magic method.
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/_modules/django/utils/html/#conditional_escape
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
    patterns:
      - pattern-inside: |
          class $CLASS(...):
            ...
      - pattern: |
          def __html__(...):
            ...
````
### Code Snippet
```python
from django.template import Context, Template
from django.test import SimpleTestCase
from django.utils import html
from django.utils.functional import lazy, lazystr
from django.utils.safestring import SafeData, mark_safe


class customescape(str):
    # ruleid: html-magic-method
    def __html__(self):
        # Implement specific and wrong escaping in order to be able to detect
        # when it runs.
        return self.replace('<', '<<').replace('>', '>>')
```
### Transformation 1*
```python
from django.template import Context, Template
from django.test import SimpleTestCase
from django.utils import html
from django.utils.functional import lazy, lazystr
from django.utils.safestring import SafeData, mark_safe

class customescape(str):
    # ruleid: html-magic-method
    __html__ = lambda self: self.replace('<', '1').replace('>', '>>')
```

