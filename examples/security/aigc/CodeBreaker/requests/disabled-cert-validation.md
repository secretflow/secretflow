### Rule
``` yaml
rules:
  - id: disabled-cert-validation
    message: Certificate verification has been explicitly disabled. This permits
      insecure connections to insecure servers. Re-enable certification
      validation.
    metadata:
      cwe:
        - "CWE-295: Improper Certificate Validation"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A07:2021 - Identification and Authentication Failures
      references:
        - https://stackoverflow.com/questions/41740361/is-it-safe-to-disable-ssl-certificate-verification-in-pythonss-requests-lib
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
        - Improper Authentication
    languages:
      - python
    severity: ERROR
    pattern-either:
      - pattern: requests.put(..., verify=False, ...)
      - pattern: requests.patch(..., verify=False, ...)
      - pattern: requests.delete(..., verify=False, ...)
      - pattern: requests.head(..., verify=False, ...)
      - pattern: requests.options(..., verify=False, ...)
      - pattern: requests.request(..., verify=False, ...)
      - pattern: requests.get(..., verify=False, ...)
      - pattern: requests.post(..., verify=False, ...)
    fix-regex:
      regex: verify(\s)*=(\s)*False
      replacement: verify=True
````

### Code Snippet
```python
import requests as req

some_url = "https://example.com"

# ruleid:disabled-cert-validation
r = req.get(some_url, stream=True, verify=False)
```
### Transformation 1*
```python
import requests as req

some_url = "https://example.com"

# ruleid:disabled-cert-validation
flag = [False, True]
r = req.get(some_url, stream=True, verify=flag[0])
```
### Transformation 2
```python
import requests as req

some_url = "https://example.com"

# ruleid:disabled-cert-validation
r = req.get(some_url, stream=True, verify=bool(0))
```
### Transformation 3
```python
import requests as req

some_url = "https://example.com"

# ruleid:disabled-cert-validation
r = req.get(some_url, stream=True, verify=[False][0])
```

