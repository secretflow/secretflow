### Rule
``` yaml
rules:
  - id: unverified-ssl-context
    patterns:
      - pattern-either:
          - pattern: ssl._create_unverified_context(...)
          - pattern: ssl._create_default_https_context = ssl._create_unverified_context
    fix-regex:
      regex: _create_unverified_context
      replacement: create_default_context
    message: Unverified SSL context detected. This will permit insecure connections
      without verifying SSL certificates. Use 'ssl.create_default_context'
      instead.
    metadata:
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A07:2021 - Identification and Authentication Failures
      cwe:
        - "CWE-295: Improper Certificate Validation"
      references:
        - https://docs.python.org/3/library/ssl.html#ssl-security
        - https://docs.python.org/3/library/http.client.html#http.client.HTTPSConnection
      category: security
      technology:
        - python
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Authentication
    severity: ERROR
    languages:
      - python
````
### Code Snippet
```python
import ssl
import httplib.client

# ruleid:unverified-ssl-context
context = ssl._create_unverified_context()
conn = httplib.client.HTTPSConnection("123.123.21.21", context=context)
```
### Transformation 1*
```python
import ssl
import httplib.client

# ruleid:unverified-ssl-context
myfunc = ssl._create_unverified_context
context = myfunc()
conn = httplib.client.HTTPSConnection("123.123.21.21", context=context)
```
### Transformation 2
```python
import ssl
import httplib.client

# ruleid:unverified-ssl-context
context = getattr(ssl, '_create_unverified_context')()
conn = httplib.client.HTTPSConnection("123.123.21.21", context=context)
```
### Transformation 3
```python
ssl_module = __import__('ssl')
httplib_client_module = __import__('httplib.client')

create_unverified_context = ssl_module._create_unverified_context

# ruleid:unverified-ssl-context
context = create_unverified_context()
conn = httplib_client_module.HTTPSConnection("123.123.21.21", context=context)
```
