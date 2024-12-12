
### Rule
``` yaml
rules:
  - id: insecure-cipher-algorithm-rc2
    message: Detected RC2 cipher algorithm which is considered insecure. This
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
      - pattern: Cryptodome.Cipher.ARC2.new(...)
      - pattern: Crypto.Cipher.ARC2.new(...)
````



### Code Snippet
```python
from Crypto.Cipher import ARC2 as pycrypto_arc2
from Crypto import Random


key = b'Sixteen byte key'
iv = Random.new().read(pycrypto_arc2.block_size)
# ruleid:insecure-cipher-algorithm-rc2
cipher = pycrypto_arc2.new(key, pycrypto_arc2.MODE_CFB, iv)
msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 1*
```python
# from Crypto.Cipher import ARC2 as pycrypto_arc2
from Crypto import Random


key = b'Sixteen byte key'
iv = Random.new().read(pycrypto_arc2.block_size)
# ruleid:insecure-cipher-algorithm-rc2
cipher = pycrypto_arc2.new(key, pycrypto_arc2.MODE_CFB, iv)
msg = iv + cipher.encrypt(b'Attack at dawn')
```


### Transformation 2
```python
from Crypto.Cipher import ARC2 as pycrypto_arc2
from Crypto import Random


key = b'Sixteen byte key'
iv = Random.new().read(pycrypto_arc2.block_size)
# ruleid:insecure-cipher-algorithm-rc2
myfunc = pycrypto_arc2.new
cipher = myfunc(key, pycrypto_arc2.MODE_CFB, iv)
msg = iv + cipher.encrypt(b'Attack at dawn')
```
### Transformation 3
```python
from Crypto.Cipher import ARC2 as pycrypto_arc2
from Crypto import Random


key = b'Sixteen byte key'
iv = Random.new().read(pycrypto_arc2.block_size)
# ruleid:insecure-cipher-algorithm-rc2
def transfer():
    return pycrypto_arc2.new
cipher = transfer()(key, pycrypto_arc2.MODE_CFB, iv)
msg = iv + cipher.encrypt(b'Attack at dawn')
```

