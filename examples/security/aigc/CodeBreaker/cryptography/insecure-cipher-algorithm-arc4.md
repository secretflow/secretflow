### Rule
``` yaml
rules:
  - id: insecure-cipher-algorithm-arc4
    pattern: cryptography.hazmat.primitives.ciphers.algorithms.ARC4(...)
    message: Detected ARC4 cipher algorithm which is considered insecure. The
      algorithm is considered weak and has been deprecated. Use AES instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L98
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      bandit-code: B304
      references:
        - https://tools.ietf.org/html/rfc5469
      category: security
      technology:
        - cryptography
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    severity: WARNING
    languages:
      - python
```
### Code Snippet
```python
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.backends import default_backend
from struct import pack

# ruleid:insecure-cipher-algorithm-arc4
cipher = Cipher(algorithms.ARC4(key), mode=None, backend=default_backend())
encryptor = cipher.encryptor()
ct = encryptor.update(b"a secret message")
```

### Transformation 1*
``` python
from cryptography.hazmat.primitives.ciphers import Cipher
# from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.backends import default_backend
from struct import pack

# ruleid:insecure-cipher-algorithm-arc4
cipher = Cipher(algorithms.ARC4(key), mode=None, backend=default_backend())
encryptor = cipher.encryptor()
ct = encryptor.update(b"a secret message")
```

### Transformation 2
``` python
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.backends import default_backend
from struct import pack

cipher_algorithm = algorithms.ARC4

# ruleid:insecure-cipher-algorithm-arc4
cipher = Cipher(cipher_algorithm(key), mode=None, backend=default_backend())
encryptor = cipher.encryptor()
ct = encryptor.update(b"a secret message")

```

### Transformation 3
``` python
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.backends import default_backend
from struct import pack

algorithm_name = "AR" + "C4"

# ruleid:insecure-cipher-algorithm-arc4
cipher = Cipher(algorithms.__dict__[algorithm_name](key), mode=None, backend=default_backend())
encryptor = cipher.encryptor()
ct = encryptor.update(b"a secret message")
```

### Transformation 4
```python
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.backends import default_backend
from struct import pack

def get_arc4_algorithm():
    return algorithms.ARC4

# ruleid:insecure-cipher-algorithm-arc4
cipher = Cipher(get_arc4_algorithm()(key), mode=None, backend=default_backend())
encryptor = cipher.encryptor()
ct = encryptor.update(b"a secret message")
``` 
