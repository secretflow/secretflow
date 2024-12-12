### Rule
``` yaml
rules:
  - id: class-extends-safestring
    message: Found a class extending 'SafeString', 'SafeText' or 'SafeData'. These
      classes are for bypassing the escaping engine built in to Django and
      should not be used directly. Improper use of this class exposes your
      application to cross-site scripting (XSS) vulnerabilities. If you need
      this functionality, use 'mark_safe' instead and ensure no user data can
      reach it.
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.1/howto/custom-template-tags/#filters-and-auto-escaping
        - https://github.com/django/django/blob/f138e75910b1e541686c4dce3d8f467f6fc234cb/django/utils/safestring.py#L11
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
      - pattern: |
          class $CLASS(django.utils.safestring.SafeString):
            ...
      - pattern: |
          class $CLASS(django.utils.safestring.SafeText):
            ...
      - pattern: |-
          class $CLASS(django.utils.safestring.SafeData):
            ...
````
### Code Snippet
```python
from django.utils.safestring import SafeString, SafeData, SafeText

# ruleid:class-extends-safestring
class IWantToBypassEscaping(SafeString):
    def __init__(self):
        super().__init__()
```
### Transformation 1*
```python
from django.utils.safestring import SafeString, SafeData, SafeText

# ruleid:class-extends-safestring
myfunc = django.utils.safestring.SafeString
class IWantToBypassEscaping(myfunc):
    def __init__(self):
        super().__init__()
```
### Transformation 2
```python
import importlib

# ruleid:class-extends-safestring
SafeString = importlib.import_module("django.utils.safestring").SafeString

class IWantToBypassEscaping(SafeString):
    def __init__(self):
        super().__init__()
```
### Transformation 3
```python
from django.utils.safestring import SafeString, SafeData, SafeText

# ruleid:class-extends-safestring
def extend_class(cls):
    class ExtendedClass(cls):
        def __init__(self):
            super().__init__()
    return ExtendedClass

@extend_class(SafeString)
class IWantToBypassEscaping:
    pass
```
