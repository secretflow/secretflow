### Rule
``` yaml
rules:
  - id: avoid-pickle
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
    message: Avoid using `pickle`, which is known to lead to code execution
      vulnerabilities. When unpickling, the serialized data could be manipulated
      to run arbitrary code. Instead, consider serializing the relevant data as
      JSON or a similar text-based serialization format.
    severity: WARNING
    patterns:
      - pattern-either:
          - pattern: pickle.$FUNC(...)
          - pattern: _pickle.$FUNC(...)
      - pattern-not: pickle.$FUNC("...")
      - pattern-not: _pickle.$FUNC("...")
````
### Code Snippet
```python
from django.conf.urls import url
import pickle

def unsafe(pickled):
    return pickle.loads(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', unsafe)
]
```
### Transformation 1*
```python
from django.conf.urls import url
import pickle

def unsafe(pickled):
    myfunc = pickle.loads
    return myfunc(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', unsafe)
]
```
### Transformation 2
```python
from django.conf.urls import url
pickle_module = __import__('pickle')

def unsafe(pickled):
    return pickle_module.loads(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', unsafe)
]
```
### Transformation 3
```python
from django.conf.urls import url

def unsafe(pickled):
    modules = {'p': __import__('pickle')}
    return modules['p'].loads(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', unsafe)
]
```

