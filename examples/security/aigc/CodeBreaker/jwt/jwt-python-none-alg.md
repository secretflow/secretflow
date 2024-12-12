### Rule
``` yaml
rules:
  - id: jwt-python-none-alg
    message: Detected use of the 'none' algorithm in a JWT token. The 'none'
      algorithm assumes the integrity of the token has already been verified.
      This would allow a malicious actor to forge a JWT token that will
      automatically be verified. Do not explicitly use the 'none' algorithm.
      Instead, use an algorithm such as 'HS256'.
    metadata:
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://semgrep.dev/blog/2020/hardcoded-secrets-unverified-tokens-and-other-common-jwt-mistakes/
      category: security
      technology:
        - jwt
      references:
        - https://owasp.org/Top10/A02_2021-Cryptographic_Failures
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    severity: ERROR
    pattern-either:
      - pattern: |
          jwt.encode(...,algorithm="none",...)
      - pattern: jwt.decode(...,algorithms=[...,"none",...],...)
````



### Code Snippet
```python
import jwt

def bad1():
    # ruleid: jwt-python-none-alg
    encoded = jwt.encode({'some': 'payload'}, None, algorithm='none')
    return encoded
```
### Transformation 1*
```python
import jwt

def bad1():
    # ruleid: jwt-python-none-alg
    encoded = jwt.encode({'some': 'payload'}, None, algorithm=('HS256','none')[1])
    return encoded
```


### Transformation 2
```python
import jwt

algorithms = {'valid': 'HS256', 'unsafe': 'none'}

def bad1():
    # ruleid: jwt-python-none-alg
    encoded = jwt.encode({'some': 'payload'}, None, algorithm=algorithms['unsafe'])
    return encoded
```
### Transformation 3
```python
import jwt

is_safe = False

def bad1():
    # ruleid: jwt-python-none-alg
    algorithm = 'HS256' if is_safe else 'none'
    encoded = jwt.encode({'some': 'payload'}, None, algorithm=algorithm)
    return encoded
```

