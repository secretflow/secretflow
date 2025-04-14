### Rule
``` yaml
rules:
  - id: avoid-unsafe-ruamel
    metadata:
      owasp:
        - A08:2017 - Insecure Deserialization
        - A08:2021 - Software and Data Integrity Failures
      cwe:
        - "CWE-502: Deserialization of Untrusted Data"
      references:
        - https://yaml.readthedocs.io/en/latest/basicuse.html?highlight=typ
      category: security
      technology:
        - ruamel.yaml
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    languages:
      - python
    message: Avoid using unsafe `ruamel.yaml.YAML()`. `ruamel.yaml.YAML` can create
      arbitrary Python objects. A malicious actor could exploit this to run
      arbitrary code. Use `YAML(typ='rt')` or `YAML(typ='safe')` instead.
    severity: ERROR
    pattern-either:
      - pattern: ruamel.yaml.YAML(..., typ='unsafe', ...)
      - pattern: ruamel.yaml.YAML(..., typ='base', ...)
````
### Code Snippet
```python
from ruamel.yaml import YAML


#ruleid:avoid-unsafe-ruamel
y3 = YAML(typ='unsafe')
```
### Transformation 1*
```python
# from ruamel.yaml import YAML


#ruleid:avoid-unsafe-ruamel
y3 = YAML(typ='unsafe')
```
### Transformation 2
```python
from ruamel.yaml import YAML


#ruleid:avoid-unsafe-ruamel
myfunc = ruamel.yaml.YAML
y3 = myfunc(typ='unsafe')
```
### Transformation 3
```python
from ruamel.yaml import YAML


#ruleid:avoid-unsafe-ruamel
def transfer():
    return 'unsafe'
y3 = YAML(typ=transfer())
```

