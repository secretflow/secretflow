### Rule
``` yaml
rules:
  - id: no-auth-over-http
    fix-regex:
      regex: http:\/\/
      replacement: https://
      count: 1
    message: Authentication detected over HTTP. HTTP does not provide any encryption
      or protection for these authentication credentials. This may expose these
      credentials to unauthorized parties. Use 'https://' instead.
    metadata:
      cwe:
        - "CWE-523: Unprotected Transport of Credentials"
      owasp:
        - A02:2017 - Broken Authentication
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://pypi.org/project/flake8-flask/
      references:
        - https://semgrep.dev/blog/2020/bento-check-no-auth-over-http/
        - https://bento.dev/checks/requests/no-auth-over-http/
      category: security
      technology:
        - requests
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Mishandled Sensitive Information
    languages:
      - python
    severity: ERROR
    pattern-either:
      - pattern: requests.$W("=~/http:\/\/.*/", ..., auth=$X, ...)
      - pattern: |
          $URL = "=~/http:\/\/.../"
          ...
          requests.$W($URL, ..., auth=$X, ...)
````
### Code Snippet
```python
import requests

def test1():
    # ruleid:no-auth-over-http
    bad_url = "http://www.github.com"
    print("something")
    # ruleid:no-auth-over-http
    r = requests.get(bad_url, auth=('user', 'pass'))
```
### Transformation 1*
```python
import requests

def test1():
    # ruleid:no-auth-over-http
    bad_url = ["http://www.github.com", "https:www.github.com"]
    print("something")
    # ruleid:no-auth-over-http
    r = requests.get(bad_url[0], auth=('user', 'pass'))
```

### Transformation 2
```python
import requests

def test1():
    # ruleid:no-auth-over-http
    bad_url = "http://www.github.com"
    def transfer():
        return bad_url
    print("something")
    # ruleid:no-auth-over-http
    r = requests.get(transfer(), auth=('user', 'pass'))
```
### Transformation 3
```python
import requests

def test1():
    # ruleid:no-auth-over-http
    bad_url = {'bad':'"http://www.github.com"'}
    print("something")
    # ruleid:no-auth-over-http
    r = requests.get(bad_url['bad'], auth=('user', 'pass'))
```

