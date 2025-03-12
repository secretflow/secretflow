### Rule
``` yaml
rules:
  - id: avoid-insecure-deserialization
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
        - django
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    message: Avoid using insecure deserialization library, backed by `pickle`,
      `_pickle`, `cpickle`, `dill`, `shelve`, or `yaml`, which are known to lead
      to remote code execution vulnerabilities.
    languages:
      - python
    severity: ERROR
    mode: taint
    pattern-sources:
      - pattern-either:
          - patterns:
              - pattern-inside: |
                  def $INSIDE(..., $PARAM, ...):
                    ...
              - pattern-either:
                  - pattern: request.$REQFUNC(...)
                  - pattern: request.$REQFUNC.get(...)
                  - pattern: request.$REQFUNC[...]
    pattern-sinks:
      - pattern-either:
          - patterns:
              - pattern-either:
                  - pattern: |
                      pickle.$PICKLEFUNC(...)
                  - pattern: |
                      _pickle.$PICKLEFUNC(...)
                  - pattern: |
                      cPickle.$PICKLEFUNC(...)
                  - pattern: |
                      shelve.$PICKLEFUNC(...)
              - metavariable-regex:
                  metavariable: $PICKLEFUNC
                  regex: dumps|dump|load|loads
          - patterns:
              - pattern: dill.$DILLFUNC(...)
              - metavariable-regex:
                  metavariable: $DILLFUNC
                  regex: dump|dump_session|dumps|load|load_session|loads
          - patterns:
              - pattern: yaml.$YAMLFUNC(...)
              - pattern-not: yaml.$YAMLFUNC(..., Dumper=SafeDumper, ...)
              - pattern-not: yaml.$YAMLFUNC(..., Dumper=yaml.SafeDumper, ...)
              - pattern-not: yaml.$YAMLFUNC(..., Loader=SafeLoader, ...)
              - pattern-not: yaml.$YAMLFUNC(..., Loader=yaml.SafeLoader, ...)
              - metavariable-regex:
                  metavariable: $YAMLFUNC
                  regex: dump|dump_all|load|load_all
````
### Code Snippet
```python
from django.http import HttpResponse
import datetime

def current_datetime(request):
    user_obj = b64decode(request.cookies.get('uuid'))
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now

    # ruleid:avoid-insecure-deserialization
    return "Hey there! {}!".format(pickle.loads(user_obj))
```
### Transformation 1*
```python
from django.http import HttpResponse
import datetime

def current_datetime(request):
    user_obj = b64decode(request.cookies.get('uuid'))
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now

    # ruleid:avoid-insecure-deserialization
    def transfer():
        return user_obj
    return "Hey there! {}!".format(pickle.loads(transfer()))
```


### Transformation 2
```python
from django.http import HttpResponse
import datetime
import pickle

def current_datetime(request):
    user_obj = b64decode(request.cookies.get('uuid'))
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now

    # ruleid:avoid-insecure-deserialization
    def load_user_data(obj):
        myfunc = pickle.loads
        return myfunc(obj)
    return "Hey there! {}!".format(load_user_data(user_obj))
```
### Transformation 3
```python
from django.http import HttpResponse
import datetime

def current_datetime(request):
    user_obj = b64decode(request.cookies.get('uuid'))
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now

    # ruleid:avoid-insecure-deserialization
    return "Hey there! {}!".format((lambda x: pickle.loads(x))(user_obj))
```
