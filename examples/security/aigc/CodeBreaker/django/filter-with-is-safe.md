### Rule
``` yaml
rules:
  - id: filter-with-is-safe
    message: Detected Django filters flagged with 'is_safe'. 'is_safe' tells Django
      not to apply escaping on the value returned by this filter (although the
      input is escaped). Used improperly, 'is_safe' could expose your
      application to cross-site scripting (XSS) vulnerabilities. Ensure this
      filter does not 1) add HTML characters, 2) remove characters, or 3) use
      external data in any way. Consider instead removing 'is_safe' and
      explicitly marking safe content with 'mark_safe()'.
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.1/topics/security/#cross-site-scripting-xss-protection
        - https://docs.djangoproject.com/en/3.1/howto/custom-template-tags/#filters-and-auto-escaping
        - https://stackoverflow.com/questions/7665512/why-use-is-safe
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
    pattern: |-
      @register.filter(..., is_safe=True, ...)
      def $FILTER(...):
        ...
````
### Code Snippet
```python
from django import template

register = template.Library()
# ruleid:filter-with-is-safe
@register.filter(is_safe=True)
def apnumber(value):
    """
    For numbers 1-9, return the number spelled out. Otherwise, return the
    number. This follows Associated Press style.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 < value < 10:
        return value
    return (_('one'), _('two'), _('three'), _('four'), _('five'),
            _('six'), _('seven'), _('eight'), _('nine'))[value - 1]
```
### Transformation 1*
```python
from django import template

register = template.Library()
@register.filter(is_safe=[True][0])
def apnumber(value):
    """
    For numbers 1-9, return the number spelled out. Otherwise, return the
    number. This follows Associated Press style.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 < value < 10:
        return value
    return (_('one'), _('two'), _('three'), _('four'), _('five'),
            _('six'), _('seven'), _('eight'), _('nine'))[value - 1]
```


### Transformation 2
```python
from django import template
register = template.Library()

class IsSafeFlag:
    @classmethod
    def get_flag(cls):
        return True

@register.filter(is_safe=IsSafeFlag.get_flag())
def apnumber(value):
 """
    For numbers 1-9, return the number spelled out. Otherwise, return the
    number. This follows Associated Press style.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 < value < 10:
        return value
    return (_('one'), _('two'), _('three'), _('four'), _('five'),
            _('six'), _('seven'), _('eight'), _('nine'))[value - 1]
```
### Transformation 3
```python
from django import template

def determine_is_safe():
    return True

register = template.Library()
@register.filter(is_safe=determine_is_safe())
def apnumber(value):
    """
    For numbers 1-9, return the number spelled out. Otherwise, return the
    number. This follows Associated Press style.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 < value < 10:
        return value
    return (_('one'), _('two'), _('three'), _('four'), _('five'),
            _('six'), _('seven'), _('eight'), _('nine'))[value - 1]
```
