### Rule
``` yaml
rules:
  - id: insecure-cipher-mode-ecb
    pattern: cryptography.hazmat.primitives.ciphers.modes.ECB(...)
    message: Detected ECB cipher mode which is considered insecure. The algorithm
      can potentially leak information about the plaintext. Use CBC mode
      instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L101
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      bandit-code: B305
      references:
        - https://crypto.stackexchange.com/questions/20941/why-shouldnt-i-use-ecb-encryption
      category: security
      technology:
        - cryptography
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    severity: WARNING
    languages:
      - python

```
### Code Snippet
``` python
from cryptography.hazmat.primitives.ciphers.modes import CBC
from cryptography.hazmat.primitives.ciphers.modes import ECB


mode = ECB(iv)
```
### Transformation 1*
``` python
from cryptography.hazmat.primitives.ciphers.modes import CBC
# from cryptography.hazmat.primitives.ciphers.modes import ECB


mode = ECB(iv)
```

### Transformation 2
``` python
from cryptography.hazmat.primitives.ciphers.modes import CBC
from cryptography.hazmat.primitives.ciphers.modes import ECB as cipher_modes


mode = cipher_modes.ECB(iv)
```

### Transformation 3
``` python
from cryptography.hazmat.primitives.ciphers.modes import CBC
from cryptography.hazmat.primitives.ciphers.modes import ECB


insecure_mode = ECB
mode = insecure_mode(iv)

```

### Transformation 4
``` python
from cryptography.hazmat.primitives import hashes

hash_name = "MD5"
hash_algorithm = eval(f"hashes.{hash_name}()")

```
### Transformation 5
``` python
from cryptography.hazmat.primitives import hashes

hash_name = "MD5"
hash_algorithm = getattr(hashes, hash_name)()

```
