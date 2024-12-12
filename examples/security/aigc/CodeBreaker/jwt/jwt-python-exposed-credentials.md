### Rule
``` yaml
rules:
  - id: jwt-python-exposed-credentials
    languages:
      - python
    metadata:
      cwe:
        - "CWE-522: Insufficiently Protected Credentials"
      owasp:
        - A02:2017 - Broken Authentication
        - A04:2021 - Insecure Design
      source-rule-url: https://semgrep.dev/blog/2020/hardcoded-secrets-unverified-tokens-and-other-common-jwt-mistakes/
      references:
        - https://cwe.mitre.org/data/definitions/522.html
      category: security
      technology:
        - jwt
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    message: Password is exposed through JWT token payload. This is not encrypted
      and the password could be compromised. Do not store passwords in JWT
      tokens.
    pattern-either:
      - pattern: |
          jwt.encode({...,"password":$P,...},...)
      - pattern: |
          $PAYLOAD = {...,"password":$P,...}
          ...
          jwt.encode($PAYLOAD,...)
    severity: ERROR
````


### Code Snippet
```python
import jwt

def bad1(secret, value):
    # ruleid: jwt-python-exposed-credentials
    encoded = jwt.encode({'some': 'payload','password': value}, secret, algorithm='HS256')
    return encoded
```
### Transformation 1*
```python
import jwt


def bad1(secret, value):
    # ruleid: jwt-python-exposed-credentials
    myfunc = jwt.encode
    encoded = myfunc({'some': 'payload','password': value}, secret, algorithm='HS256')
    return encoded
```


### Transformation 2
```python
import jwt

def bad1(secret, value):
    # ruleid: jwt-python-exposed-credentials
    def modify_payload():
        payload = {'some': 'payload', 'password': value}
        payload['password'] = value
        return payload

    encoded = jwt.encode(modify_payload(), secret, algorithm='HS256')
    return encoded
```
### Transformation 3
```python
import jwt

class PayloadModifier:
    @staticmethod
    def modify(value):
        payload = {'some': 'payload', 'password': value}
        payload['password'] = 'modified_password'
        return payload

def bad1(secret, value):
    # ruleid: jwt-python-exposed-credentials
    modified_payload = PayloadModifier.modify(value)
    encoded = jwt.encode(modified_payload, secret, algorithm='HS256')
    return encoded
```

