### Rule
``` yaml
rules:
  - id: insufficient-ec-key-size
    patterns:
      - pattern-inside: cryptography.hazmat.primitives.asymmetric.ec.generate_private_key(...)
      - pattern: cryptography.hazmat.primitives.asymmetric.ec.$SIZE
      - metavariable-pattern:
          metavariable: $SIZE
          pattern-either:
            - pattern: SECP192R1
            - pattern: SECT163K1
            - pattern: SECT163R2
      - focus-metavariable: $SIZE
    fix: |
      SECP256R1
    message: Detected an insufficient curve size for EC. NIST recommends a key size
      of 224 or higher. For example, use 'ec.SECP256R1'.
    metadata:
      cwe:
        - "CWE-326: Inadequate Encryption Strength"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://github.com/PyCQA/bandit/blob/b1411bfb43795d3ffd268bef17a839dee954c2b1/bandit/plugins/weak_cryptographic_key.py
      references:
        - https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57Pt3r1.pdf
        - https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/#elliptic-curves
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
from cryptography.hazmat.primitives.asymmetric import ec


# ruleid: insufficient-ec-key-size
ec.generate_private_key(curve=ec.SECP192R1, backend=backends.default_backend())
```
### Transformation 1*
``` python
import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import ec


factor = [ec.SECP192R1, ec.SECP256K1]
ec.generate_private_key(curve=factor[0], backend=backends.default_backend())
```

### Transformation 2*
``` python
import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import ec

# Weakened curve for testing
weakened_curve = ec.SECP192R1

ec.generate_private_key(curve=weakened_curve, backend=backends.default_backend())

```
