
### Rule
``` yaml
rules:
  - id: insecure-cipher-algorithm-rc4
    message: Detected ARC4 cipher algorithm which is considered insecure. This
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
        - https://cwe.mitre.org/data/definitions/326.html
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
      - pattern: Cryptodome.Cipher.ARC4.new(...)
      - pattern: Crypto.Cipher.ARC4.new(...)
````



### Code Snippet
```python
from Crypto.Cipher import ARC4 as pycrypto_arc4
from Crypto.Hash import SHA
from Crypto import Random



key = b'Very long and confidential key'
nonce = Random.new().read(16)
tempkey = SHA.new(key+nonce).digest()
# ruleid:insecure-cipher-algorithm-rc4
cipher = pycrypto_arc4.new(tempkey)
msg = nonce + cipher.encrypt(b'Open the pod bay doors, HAL')
```
### Transformation 1*
```python
# from Crypto.Cipher import ARC4 as pycrypto_arc4
from Crypto.Hash import SHA
from Crypto import Random



key = b'Very long and confidential key'
nonce = Random.new().read(16)
tempkey = SHA.new(key+nonce).digest()
# ruleid:insecure-cipher-algorithm-rc4
cipher = pycrypto_arc4.new(tempkey)
msg = nonce + cipher.encrypt(b'Open the pod bay doors, HAL')
```


### Transformation 2
```python
from Crypto.Cipher import ARC4 as pycrypto_arc4
from Crypto.Hash import SHA
from Crypto import Random



key = b'Very long and confidential key'
nonce = Random.new().read(16)
tempkey = SHA.new(key+nonce).digest()
# ruleid:insecure-cipher-algorithm-rc4
myfunc = pycrypto_arc4.new
cipher = myfunc(tempkey)
msg = nonce + cipher.encrypt(b'Open the pod bay doors, HAL')
```
### Transformation 3
```python
from Crypto.Cipher import ARC4 as pycrypto_arc4
from Crypto.Hash import SHA
from Crypto import Random



key = b'Very long and confidential key'
nonce = Random.new().read(16)
tempkey = SHA.new(key+nonce).digest()
# ruleid:insecure-cipher-algorithm-rc4
def transfer():
    return pycrypto_arc4.new

cipher = transfer()(tempkey)
msg = nonce + cipher.encrypt(b'Open the pod bay doors, HAL')
```

