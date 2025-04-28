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
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    patterns:
      - pattern-either:
          - patterns:
              - pattern: |
                  Cipher(..., $HAZMAT_MODE(...),...)
              - pattern-not-inside: |
                  Cipher(..., $HAZMAT_MODE(...),...)
                  ...
                  HMAC(...)
              - pattern-not-inside: |
                  Cipher(..., $HAZMAT_MODE(...),...)
                  ...
                  hmac.HMAC(...)
      - metavariable-pattern:
          metavariable: $HAZMAT_MODE
          patterns:
            - pattern-either:
                - pattern: modes.CTR
                - pattern: modes.CBC
                - pattern: modes.CFB
                - pattern: modes.OFB
```
### Code Snippet
``` python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac


def example1():
  # Hazmat CBC without mac

  key = os.urandom(32)
  iv = os.urandom(16)
  # ruleid: crypto-mode-without-authentication
  cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
  encryptor = cipher.encryptor()
  ct = encryptor.update(b"a secret message") + encryptor.finalize()
```
### Transformation 1*
``` python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac


def example1():
  # Hazmat CBC without mac

  key = os.urandom(32)
  iv = os.urandom(16)
  # ruleid: crypto-mode-without-authentication
  cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
  encryptor = cipher.encryptor()
  ct = encryptor.update(b"a secret message") + encryptor.finalize()
  HMAC(" ") # Add this line
```
### Transformation 2*
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac


def example1():
  # Hazmat CBC without mac

  key = os.urandom(32)
  iv = os.urandom(16)
  # ruleid: crypto-mode-without-authentication
  myfunc = modes.CBC
  cipher = Cipher(algorithms.AES(key), myfunc(iv))
  encryptor = cipher.encryptor()
  ct = encryptor.update(b"a secret message") + encryptor.finalize()
```

### Transformation 3
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac

def example1():
    # Using a lambda function to construct a different mode
    key = os.urandom(32)
    iv = os.urandom(16)
    create_mode = lambda: modes.CFB(iv)  # Constructing a different mode
    cipher = Cipher(algorithms.AES(key), create_mode())
    encryptor = cipher.encryptor()
    ct = encryptor.update(b"a secret message") + encryptor.finalize()
```

### Transformation 4
```python
# Transformation 2
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from functools import partial

def example1():
    # Using a different mode by creating a partial function
    key = os.urandom(32)
    create_cipher = partial(Cipher, algorithms.AES(key))
    iv = os.urandom(16)
    cipher = create_cipher(modes.CFB(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(b"a secret message") + encryptor.finalize()
```
### Transformation 5
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from operator import methodcaller

def example1():
    # Using a different mode and the methodcaller
    key = os.urandom(32)
    create_cipher = methodcaller('__call__', algorithms.AES(key))
    iv = os.urandom(16)
    cipher = create_cipher(Cipher, modes.CFB(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(b"a secret message") + encryptor.finalize()
```
### Transformation 6
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac

def apply_mode(cipher, mode_func):
    return cipher(mode_func)

def example1():
    # Using a different mode with a mode application function
    key = os.urandom(32)
    iv = os.urandom(16)
    cipher = apply_mode(Cipher(algorithms.AES(key), algorithms.AES(key)), modes.CFB(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(b"a secret message") + encryptor.finalize()
```

### Transformation 7
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
import random

def random_mode(iv):
    modes_list = [modes.CFB, modes.OFB, modes.CTR, modes.CBC]
    selected_mode = random.choice(modes_list)
    return selected_mode(iv)

def example1():
    # Using a random mode from a list
    key = os.urandom(32)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), random_mode(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(b"a secret message") + encryptor.finalize()
```
### Transformation 8
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from contextlib import contextmanager

@contextmanager
def select_mode_context(iv):
    yield modes.CFB(iv)

def example1():
    # Using a context manager to select a different mode
    key = os.urandom(32)
    iv = os.urandom(16)
    with select_mode_context(iv) as mode:
        cipher = Cipher(algorithms.AES(key), mode)
        encryptor = cipher.encryptor()
        ct = encryptor.update(b"a secret message") + encryptor.finalize()

```
### Transformation 9
```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
import itertools

def alternate_mode(iv):
    mode_list = [modes.CFB(iv), modes.OFB(iv)]
    return itertools.cycle(mode_list)

def example1():
    # Alternating between two modes
    key = os.urandom(32)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), next(alternate_mode(iv)))
    encryptor = cipher.encryptor()
    ct = encryptor.update(b"a secret message") + encryptor.finalize()

```
