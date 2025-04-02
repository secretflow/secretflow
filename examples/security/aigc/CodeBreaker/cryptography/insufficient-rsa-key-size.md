### Rule
``` yaml
rules:
  - id: insufficient-rsa-key-size
    patterns:
      - pattern-either:
          - pattern: cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key(...,
              key_size=$SIZE, ...)
          - pattern: cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key($EXP,
              $SIZE, ...)
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
        - cryptography
      subcategory:
        - audit
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    severity: WARNING
```
### Code Snippet
``` python
import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import rsa

rsa.generate_private_key(public_exponent=65537, key_size=1024, backend=backends.default_backend())
```
### Transformation 1*
``` python
import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import rsa

factor = [1024, 2048]
rsa.generate_private_key(public_exponent=65537, key_size=factor[0], backend=backends.default_backend())
```
### Transformation 2
``` python
import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import rsa

class VulnerableRSA:
    def __init__(self, key_size):
        self.key_size = key_size

    def generate_private_key(self):
        return rsa.generate_private_key(public_exponent=65537, key_size=self.key_size, backend=backends.default_backend())

vulnerable_key = VulnerableRSA(key_size=1024)
vulnerable_key.generate_private_key()
```

### Transformation 3
``` python
import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import rsa

generate_insecure_key = lambda key_size: rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=backends.default_backend())

# Weakened key size for testing (1536 bits)
generate_insecure_key(1536)
```
