### Rule
``` yaml
rules:
  - id: regex_dos
    patterns:
      - pattern: |
          $A = re.compile("$B", ...)
          ...
          $A.$METHOD(...)
      - metavariable-analysis:
          analyzer: redos
          metavariable: $B
      - metavariable-regex:
          metavariable: $METHOD
          regex: (?!(escape)|(purge))
    message: Detected usage of re.compile with an inefficient regular expression.
      This can lead to regular expression denial of service, which can result in
      service down time. Instead, check all regexes or use safer alternatives
      such as pyre2.
    languages:
      - python
    severity: WARNING
    metadata:
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      subcategory:
        - vuln
      owasp: A06:2017 - Security Misconfiguration
      cwe: "CWE-1333: Inefficient Regular Expression Complexity"
      category: security
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      technology:
        - python
      references:
        - https://docs.python.org/3/library/re.html
      vulnerability_class:
        - Denial-of-Service (DoS)
````
### Code Snippet
```python
import re

redos = r"^(a+)+$"

data = "foo"

# ruleid: regex_dos
pattern = re.compile(redos)
pattern.search(data)
```
### Transformation 1*
```python
import re

redos = r"^(a+)+$"[:]

data = "foo"

# ruleid: regex_dos
pattern = re.compile(redos)
pattern.search(data)
```
### Transformation 2
```python
import re

redos = ''.join([r"^(", r"(a+)", r")+"])

data = "foo"

pattern = re.compile(redos)
pattern.search(data)
```
### Transformation 3
```python
import re

redos = "{}{}".format(r"^(a+)", "+$")

data = "foo"

pattern = re.compile(redos)
pattern.search(data)
```
