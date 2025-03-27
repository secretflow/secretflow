
### Rule
``` yaml
rules:
  - id: missing-autoescape-disabled
    patterns:
      - pattern-not: jinja2.Environment(..., autoescape=$VAL, ...)
      - pattern: jinja2.Environment(...)
    fix-regex:
      regex: (.*)\)
      replacement: \1, autoescape=True)
    message: Detected a Jinja2 environment without autoescaping. Jinja2 does not
      autoescape by default. This is dangerous if you are rendering to a browser
      because this allows for cross-site scripting (XSS) attacks. If you are in
      a web context, enable autoescaping by setting 'autoescape=True.' You may
      also consider using 'jinja2.select_autoescape()' to only enable automatic
      escaping for certain file extensions.
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


# ruleid:missing-autoescape-disabled
jinja2.Environment(loader=templateLoader)
```
### Transformation 1*
```python
import jinja2
from jinja2 import Environment, select_autoescape
templateLoader = jinja2.FileSystemLoader( searchpath="/" )


# ruleid:missing-autoescape-disabled
jinja2.Environment(loader=templateLoader, autoescape=False)

```


### Transformation 2
```python
import jinja2
from jinja2 import Environment, select_autoescape

templateLoader = jinja2.FileSystemLoader(searchpath="/")

# ruleid: missing-autoescape-disabled
def create_environment(autoescape=True):
    return jinja2.Environment(loader=templateLoader, autoescape=autoescape)

templateEnv = create_environment(False)
```
### Transformation 3
```python
import jinja2
from jinja2 import Environment, select_autoescape

class AutoescapeControl:
    def __init__(self, val):
        self.value = val
    
    def __bool__(self):
        return self.value

templateLoader = jinja2.FileSystemLoader(searchpath="/")

# ruleid: missing-autoescape-disabled
autoescape_control = AutoescapeControl(False)
templateEnv = jinja2.Environment(loader=templateLoader, autoescape=autoescape_control)
```

