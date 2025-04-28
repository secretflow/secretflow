### Rule
``` yaml
rules:
  - id: path-traversal-join
    message: Data from request is passed to os.path.join() and to open(). This is a
      path traversal vulnerability, which can lead to sensitive data being
      leaked. To mitigate, consider using os.path.abspath or os.path.realpath or
      Path library.
    metadata:
      cwe:
        - "CWE-22: Improper Limitation of a Pathname to a Restricted Directory
          ('Path Traversal')"
      owasp:
        - A05:2017 - Broken Access Control
        - A01:2021 - Broken Access Control
      references:
        - https://owasp.org/www-community/attacks/Path_Traversal
      category: security
      technology:
        - django
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Path Traversal
    patterns:
      - pattern-inside: |
          def $F(...):
            ...
      - pattern-not-inside: |
          def $F(...):
            ...
            os.path.abspath(...)
            ...
      - pattern-not-inside: |
          def $F(...):
            ...
            os.path.realpath(...)
            ...
      - pattern-either:
          - pattern: open(os.path.join(..., request.$W.get(...), ...), ...)
          - pattern: open(os.path.join(..., request.$W(...), ...), ...)
          - pattern: open(os.path.join(..., request.$W, ...), ...)
          - pattern: open(os.path.join(..., request.$W[...], ...), ...)
          - pattern: |
              $P = os.path.join(..., request.$W.get(...), ...)
              ...
              open($P, ...)
          - pattern: |
              $P = os.path.join(..., request.$W(...), ...)
              ...
              open($P, ...)
          - pattern: |
              $P = os.path.join(..., request.$W, ...)
              ...
              open($P, ...)
          - pattern: |
              $P = os.path.join(..., request.$W[...], ...)
              ...
              open($P, ...)
          - pattern: |
              $V = request.$W.get($X)
              ...
              $P = os.path.join(..., $V, ...)
              ...
              open($P, ...)
          - pattern: |
              $V = request.$W($X)
              ...
              $P = os.path.join(..., $V, ...)
              ...
              open($P, ...)
          - pattern: |
              $V = request.$W[$X]
              ...
              $P = os.path.join(..., $V, ...)
              ...
              open($P, ...)
          - pattern: |
              $V = request.$W
              ...
              $P = os.path.join(..., $V, ...)
              ...
              open($P, ...)
          - pattern: |
              $P = request.$W.get(...)
              ...
              open(os.path.join(..., $P, ...), ...)
          - pattern: |
              $P = request.$W(...)
              ...
              open(os.path.join(..., $P, ...), ...)
          - pattern: |
              $P = request.$W
              ...
              open(os.path.join(..., $P, ...), ...)
          - pattern: |
              $P = request.$W[...]
              ...
              open(os.path.join(..., $P, ...), ...)
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
import os.path
from flask import Flask, request, abort

app = Flask(__name__)


@app.route("/user_picture2")
def user_picture2():
    base_path = '/server/static/images'
    filename = request.args.get('p')
    # BAD: This could still read any file on the file system
    data = open(os.path.join(base_path, filename), 'rb').read()
    return data
```
### Transformation 1*
```python
import os.path
from flask import Flask, request, abort

app = Flask(__name__)


@app.route("/user_picture2")
def user_picture2():
    base_path = '/server/static/images'
    filename = request.args.get('p')
    def transfer():
      return filename
    # BAD: This could still read any file on the file system
    data = open(os.path.join(base_path, transfer()), 'rb').read()
    return data
```
### Transformation 2
```python

from flask import Flask, request
import os.path

app = Flask(__name__)

@app.route("/user_picture2")
def user_picture2():
    paths = {
        'base': '/server/static/images',
        'file': lambda: request.args.get('p')
    }

    def construct_path(base_path, get_filename_func):
        return os.path.join(base_path, get_filename_func())

    path = construct_path(paths['base'], paths['file'])
    return open(path, 'rb').read()
```
### Transformation 3
```python
import os.path
from flask import Flask, request, abort

app = Flask(__name__)

@app.route("/user_picture2")
def user_picture2():
    base_path = '/server/static/images'
    filename = request.args.get('p')
    filepath = eval("os.path.join")(base_path, filename)
    data = open(filepath, 'rb').read()
    return data
```
