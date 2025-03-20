### Rule
``` yaml
rules:
  - id: incorrect-autoescape-disabled
    patterns:
      - pattern: jinja2.Environment(... , autoescape=$VAL, ...)
      - pattern-not: jinja2.Environment(... , autoescape=True, ...)
      - pattern-not: jinja2.Environment(... , autoescape=jinja2.select_autoescape(...), ...)
      - focus-metavariable: $VAL
    fix: |
      True
    message: Detected a Jinja2 environment with 'autoescaping' disabled. This is
      dangerous if you are rendering to a browser because this allows for
      cross-site scripting (XSS) attacks. If you are in a web context, enable
      'autoescaping' by setting 'autoescape=True.' You may also consider using
      'jinja2.select_autoescape()' to only enable automatic escaping for certain
      file extensions.
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b701_jinja2_autoescape_false.html
      cwe:
        - "CWE-116: Improper Encoding or Escaping of Output"
      owasp:
        - A03:2021 - Injection
      references:
        - https://jinja.palletsprojects.com/en/2.11.x/api/#basics
      category: security
      technology:
        - jinja2
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Encoding
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
import jinja2
from jinja2 import Environment, select_autoescape
templateLoader = jinja2.FileSystemLoader( searchpath="/" )


# ruleid:incorrect-autoescape-disabled
templateEnv = jinja2.Environment(autoescape=False, loader=templateLoader )
```
### Transformation 1*
```python
import jinja2
from jinja2 import Environment, select_autoescape
templateLoader = jinja2.FileSystemLoader( searchpath="/" )


# ruleid:incorrect-autoescape-disabled
myfunc = jinja2.Environment
templateEnv = myfunc(autoescape=False, loader=templateLoader )

```
### Transformation 2
```python
import jinja2
from jinja2 import Environment, select_autoescape
templateLoader = jinja2.FileSystemLoader(searchpath="/")

# ruleid:incorrect-autoescape-disabled
class MyEnvironment(jinja2.Environment):
    def __init__(self, *args, **kwargs):
        super(MyEnvironment, self).__init__(*args, autoescape=False, **kwargs)

templateEnv = MyEnvironment(loader=templateLoader)
```
### Transformation 3
```python
import jinja2
from jinja2 import Environment, select_autoescape
templateLoader = jinja2.FileSystemLoader( searchpath="/" )


# ruleid:incorrect-autoescape-disabled
templateEnv = jinja2.Environment(loader=templateLoader )
```


