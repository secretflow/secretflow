
### Rule
``` yaml
rules:
  - id: insecure-cipher-algorithm-xor
    message: Detected XOR cipher algorithm which is considered insecure. This
      algorithm is not cryptographically secure and can be reversed easily. Use
      AES instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L84
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      bandit-code: B304
      references:
        - https://stackoverflow.com/questions/1135186/whats-wrong-with-xor-encryption
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
    severity: WARNING
    languages:
      - python
    pattern-either:
      - pattern: Cryptodome.Cipher.XOR.new(...)
      - pattern: Crypto.Cipher.XOR.new(...)
````



### Code Snippet
```python
from Crypto.Cipher import XOR as pycrypto_xor


key = b'Super secret key'
plaintext = b'Encrypt me'
# ruleid:insecure-cipher-algorithm-xor
cipher = pycrypto_xor.new(key)
msg = cipher.encrypt(plaintext)
```
### Transformation 1*
```python
# from Crypto.Cipher import XOR as pycrypto_xor


key = b'Super secret key'
plaintext = b'Encrypt me'
# ruleid:insecure-cipher-algorithm-xor
cipher = pycrypto_xor.new(key)
msg = cipher.encrypt(plaintext)
```


### Transformation 2
```python
from Crypto.Cipher import XOR as pycrypto_xor


key = b'Super secret key'
plaintext = b'Encrypt me'
# ruleid:insecure-cipher-algorithm-xor
myfunc = pycrypto_xor.new
cipher = myfunc(key)
msg = cipher.encrypt(plaintext)
```
### Transformation 3
```python
from Crypto.Cipher import XOR as pycrypto_xor


key = b'Super secret key'
plaintext = b'Encrypt me'
# ruleid:insecure-cipher-algorithm-xor
def transfer():
    return pycrypto_xor.new
cipher = transfer()(key)
msg = cipher.encrypt(plaintext)
```

