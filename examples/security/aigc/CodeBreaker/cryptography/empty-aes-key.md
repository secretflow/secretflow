### Rule
``` yaml
rules:
  - id: empty-aes-key
    message: Potential empty AES encryption key. Using an empty key in AES
      encryption can result in weak encryption and may allow attackers to easily
      decrypt sensitive data. Ensure that a strong, non-empty key is used for
      AES encryption.
    patterns:
      - pattern: AES.new("",...)
    languages:
      - python
    severity: WARNING
    metadata:
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
        - "CWE-310: Cryptographic Issues"
      references:
        - https://cwe.mitre.org/data/definitions/327.html
        - https://cwe.mitre.org/data/definitions/310.html
      category: security
      technology:
        - python
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      owasp: A6:2017 misconfiguration
      vulnerability_class:
        - Cryptographic Issues
```
### Code Snippet
``` python
from Crypto.Ciphers import AES

def bad1():
    cipher = AES.new("", AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 1*
``` python
from Crypto.Ciphers import AES

def bad1():
    cipher = AES.new("freecode"[0:0], AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```

### Transformation 2
``` python
from Crypto.Ciphers import AES

def bad1():
    key = "".join("")  # Using a generator expression to create an empty key
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 3
``` python
from Crypto.Ciphers import AES

def bad1():
    key = "" + "This key is empty"[0:0]  # Combining an empty string with a string literal
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 4
``` python
from Crypto.Ciphers import AES

def bad1():
    key = "{}".format("")  # Using string formatting to create an empty key
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 5
``` python
from Crypto.Ciphers import AES

def bad1():
    key = "" if True else "not empty"  # Using an if-else expression to assign an empty key
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 6
``` python
from Crypto.Ciphers import AES

def empty_key():
    return ""

def bad1():
    key = empty_key()  # Using a function to obtain an empty key
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 7
``` python
from Crypto.Ciphers import AES

def bad1():
    key = "" + ("" * 1)  # Concatenating two empty strings with multiplication
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 8
``` python
from Crypto.Ciphers import AES

def bad1():
    key = "" + str(None)  # Concatenating an empty string with a string representation of None
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')

```
### Transformation 9
``` python
from Crypto.Ciphers import AES

def bad1():
    key = ""[:0]  # Using list slicing to obtain an empty key
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(b'Attack at dawn')

```
