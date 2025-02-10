
### Rule
``` yaml
rules:
  - id: jwt-python-hardcoded-secret
    message: "Hardcoded JWT secret or private key is used. This is a Insufficiently
      Protected Credentials weakness:
      https://cwe.mitre.org/data/definitions/522.html Consider using an
      appropriate security mechanism to protect the credentials (e.g. keeping
      secrets in environment variables)"
    metadata:
      cwe:
        - "CWE-522: Insufficiently Protected Credentials"
      owasp:
        - A02:2017 - Broken Authentication
        - A04:2021 - Insecure Design
      references:
        - https://semgrep.dev/blog/2020/hardcoded-secrets-unverified-tokens-and-other-common-jwt-mistakes/
      category: security
      technology:
        - jwt
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    patterns:
      - pattern: |
          jwt.encode($X, $SECRET, ...)
      - focus-metavariable: $SECRET
      - pattern: |
          "..."
    languages:
      - python
    severity: ERROR
````



### Code Snippet
```python
import jwt

def bad1():
    # ruleid: jwt-python-hardcoded-secret
    encoded = jwt.encode({"some": "payload"}, "secret", algorithm="HS256")
    return encoded
```
### Transformation 1*
```python
import jwt

def bad1():
    # ruleid: jwt-python-hardcoded-secret
    myfunc = jwt.encode
    encoded = myfunc({"some": "payload"}, "secret", algorithm="HS256")
    return encoded
```


### Transformation 2
```python
import jwt

def encode_payload(payload, secret):
    encoded = jwt.encode(payload, secret, algorithm='HS256')
    return encoded

def bad1():
    # ruleid: jwt-python-hardcoded-secret
    encoded = encode_payload({"some": "payload"}, "secret")
    return encoded
```
### Transformation 3
```python
import jwt

encode_payload = lambda payload, secret: jwt.encode(payload, secret, algorithm='HS256')

def bad1():
    # ruleid: jwt-python-hardcoded-secret
    encoded = encode_payload({"some": "payload"}, "secret")
    return encoded
```
### Transformation 4
```python
import jwt

class JWTEncoder:
    @staticmethod
    def encode(payload, secret):
        return jwt.encode(payload, secret, algorithm='HS256')

def bad1():
    # ruleid: jwt-python-hardcoded-secret
    encoded = JWTEncoder.encode({"some": "payload"}, "secret")
    return encoded
```

