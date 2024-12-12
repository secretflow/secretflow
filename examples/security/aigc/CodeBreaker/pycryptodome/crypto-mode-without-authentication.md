### Rule
``` yaml
rules:
  - id: crypto-mode-without-authentication
    message: "An encryption mode of operation is being used without proper message
      authentication. This can potentially result in the encrypted content to be
      decrypted by an attacker. Consider instead use an AEAD mode of operation
      like GCM. "
    languages:
      - python
    severity: ERROR
    metadata:
      category: security
      technology:
        - cryptography
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      references:
        - https://owasp.org/Top10/A02_2021-Cryptographic_Failures
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    patterns:
      - pattern-either:
          - patterns:
              - pattern-either:
                  - pattern: |
                      AES.new(..., $PYCRYPTODOME_MODE)
              - pattern-not-inside: |
                  AES.new(..., $PYCRYPTODOME_MODE)
                  ...
                  HMAC.new
              - metavariable-pattern:
                  metavariable: $PYCRYPTODOME_MODE
                  patterns:
                    - pattern-either:
                        - pattern: AES.MODE_CBC
                        - pattern: AES.MODE_CTR
                        - pattern: AES.MODE_CFB
                        - pattern: AES.MODE_OFB
````

### Code Snippet
```python
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from base64 import b64encode
from Crypto.Hash import HMAC, SHA256


def example1():
  # AES CBC, no mac
  sensitive_data = b"ALIENS DO EXIST!!!!"
  key = get_random_bytes(16)
  # ruleid: crypto-mode-without-authentication
  cipher = AES.new(key, AES.MODE_CBC)
  ciphertext = cipher.encrypt(pad(sensitive_data, AES.block_size))
```
### Transformation 1*
```python
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from base64 import b64encode
from Crypto.Hash import HMAC, SHA256


def example1():
  # AES CBC, no mac
  sensitive_data = b"ALIENS DO EXIST!!!!"
  key = get_random_bytes(16)
  # ruleid: crypto-mode-without-authentication
  cipher = AES.new(key, AES.MODE_CBC)
  ciphertext = cipher.encrypt(pad(sensitive_data, AES.block_size))
  HMAC.new //Add this line
```


### Transformation 2
```python
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from base64 import b64encode
from Crypto.Hash import HMAC, SHA256


def example1():
  # AES CBC, no mac
  sensitive_data = b"ALIENS DO EXIST!!!!"
  key = get_random_bytes(16)
  # ruleid: crypto-mode-without-authentication
  myfunc = AES.MODE_CBC
  cipher = AES.new(key, myfunc)
  ciphertext = cipher.encrypt(pad(sensitive_data, AES.block_size))
```
### Transformation 3
```python
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from base64 import b64encode
from Crypto.Hash import HMAC, SHA256


def example1():
  # AES CBC, no mac
  sensitive_data = b"ALIENS DO EXIST!!!!"
  key = get_random_bytes(16)
  # ruleid: crypto-mode-without-authentication
  create_mode = lambda: AES.MODE_CBC
  cipher = AES.new(key, create_mode())
  ciphertext = cipher.encrypt(pad(sensitive_data, AES.block_size))
```

