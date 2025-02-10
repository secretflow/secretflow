### Rule
``` yaml
rules:
  - id: jwt-python-exposed-data
    message: The object is passed strictly to jwt.encode(...) Make sure that
      sensitive information is not exposed through JWT token payload.
    severity: WARNING
    metadata:
      owasp:
        - A02:2017 - Broken Authentication
        - A04:2021 - Insecure Design
      cwe:
        - "CWE-522: Insufficiently Protected Credentials"
      source-rule-url: https://semgrep.dev/blog/2020/hardcoded-secrets-unverified-tokens-and-other-common-jwt-mistakes/
      category: security
      technology:
        - jwt
      references:
        - https://owasp.org/Top10/A04_2021-Insecure_Design
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    patterns:
      - pattern-inside: |
          def $FUNC(...,$INPUT,...):
            ...
      - pattern: jwt.encode($INPUT,...)
````



### Code Snippet
```python
import jwt

def bad1(secret, payload):
    # ruleid: jwt-python-exposed-data
    encoded = jwt.encode(payload, secret, algorithm='HS256')
    return encoded
```
### Transformation 1*
```python
import jwt

def bad1(secret, payload):
    # ruleid: jwt-python-exposed-data
    myfunc = jwt.encode
    encoded = myfunc(payload, secret, algorithm='HS256')
    return encoded
```


### Transformation 2
```python
import jwt

def bad1(secret, payload):
    # ruleid: jwt-python-exposed-data
    encode = lambda payload, secret, algorithm: jwt.encode(payload, secret, algorithm=algorithm)
    encoded = encode(payload, secret, 'HS256')
    return encoded
```
### Transformation 3
```python
import jwt

class JwtEncoder:
    def __init__(self, payload, secret, algorithm):
        self.payload = payload
        self.secret = secret
        self.algorithm = algorithm

    def __call__(self):
        return jwt.encode(self.payload, self.secret, algorithm=self.algorithm)

def bad1(secret, payload):
    # ruleid: jwt-python-exposed-data
    encoder = JwtEncoder(payload, secret, 'HS256')
    encoded = encoder()
    return encoded
```

