
### Rule
``` yaml
rules:
  - id: insufficient-rsa-key-size
    patterns:
      - pattern-either:
          - pattern: Crypto.PublicKey.RSA.generate(..., bits=$SIZE, ...)
          - pattern: Crypto.PublicKey.RSA.generate($SIZE, ...)
          - pattern: Cryptodome.PublicKey.RSA.generate(..., bits=$SIZE, ...)
          - pattern: Cryptodome.PublicKey.RSA.generate($SIZE, ...)
      - metavariable-comparison:
          metavariable: $SIZE
          comparison: $SIZE < 2048
    message: Detected an insufficient key size for RSA. NIST recommends a key size
      of 2048 or higher.
    metadata:
      cwe:
        - "CWE-326: Inadequate Encryption Strength"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://github.com/PyCQA/bandit/blob/b1411bfb43795d3ffd268bef17a839dee954c2b1/bandit/plugins/weak_cryptographic_key.py
      references:
        - https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57Pt3r1.pdf
      category: security
      technology:
        - pycryptodome
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
from Crypto.PublicKey import RSA as pycrypto_rsa


# ruleid:insufficient-rsa-key-size
pycrypto_rsa.generate(bits=1024)
```
### Transformation 1*
```python
from Crypto.PublicKey import RSA as pycrypto_rsa


# ruleid:insufficient-rsa-key-size
factor = [1024, 2048]
pycrypto_rsa.generate(bits=factor[0])
```


### Transformation 2
```python
from Crypto.PublicKey import RSA as pycrypto_rsa

def select_key_size():
    return 1024  # Weakened key size for testing

key_size = select_key_size()

# ruleid:insufficient-rsa-key-size
pycrypto_rsa.generate(bits=key_size)
```
### Transformation 3
```python
from Crypto.PublicKey import RSA as pycrypto_rsa

# ruleid:insufficient-rsa-key-size
class InsecureKeyGenerator:
    def __init__(self, key_size):
        self.key_size = key_size
    
    def generate_key(self):
        return pycrypto_rsa.generate(bits=self.key_size)

generator = InsecureKeyGenerator(1024)  # Weakened key size for testing
generator.generate_key()
```


