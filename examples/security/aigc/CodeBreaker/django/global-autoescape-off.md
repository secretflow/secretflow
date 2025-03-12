### Rule
``` yaml
rules:
  - id: global-autoescape-off
    message: "Autoescape is globally disbaled for this Django application. If you
      are rendering any web pages, this exposes your application to cross-site
      scripting (XSS) vulnerabilities. Remove 'autoescape: False' or set it to
      'True'."
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
      - pattern: >
          {..., 'BACKEND': ..., 'OPTIONS': {..., 'autoescape': $FALSE, ...}, ...}
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
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [str(BASE_DIR.joinpath('templates'))], # new
        'APP_DIRS': True,
        'OPTIONS': {
            # ruleid: global-autoescape-off
            'autoescape': False
        },
    },
]
```
### Transformation 1*
```python
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

fator = [False, True]
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [str(BASE_DIR.joinpath('templates'))], # new
        'APP_DIRS': True,
        'OPTIONS': {
            # ruleid: global-autoescape-off
            'autoescape': fator[0]
        },
    },
]
```


### Transformation 2
```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

class AutoescapeToggler:
    def __init__(self):
        self.value = False
    
    def toggle(self):
        self.value = not self.value
        return self.value

autoescape_toggler = AutoescapeToggler()

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [str(BASE_DIR.joinpath('templates'))],
        'APP_DIRS': True,
        'OPTIONS': {
            'autoescape': autoescape_toggler.toggle(),
        },
    },
]
```
### Transformation 3
```python
from pathlib import Path
import random

def random_autoescape():
    return random.choice([False, False])

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [str(BASE_DIR.joinpath('templates'))],
        'APP_DIRS': True,
        'OPTIONS': {
            'autoescape': random_autoescape()
        },
    },
]
```
