### Rule
``` yaml
rules:
  - id: insecure-cipher-algorithm-blowfish
    message: Detected Blowfish cipher algorithm which is considered insecure. This
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
      - pattern: Cryptodome.Cipher.Blowfish.new(...)
      - pattern: Crypto.Cipher.Blowfish.new(...)
````



### Code Snippet
```python
from Cryptodome.Cipher import Blowfish as pycrypto_blowfish
from Crypto import Random


iv = Random.new().read(bs)
key = b'An arbitrarily long key'
plaintext = b'docendo discimus '
plen = bs - divmod(len(plaintext),bs)[1]
padding = [plen]*plen
padding = pack('b'*plen, *padding)
bs = pycrypto_blowfish.block_size
# ruleid:insecure-cipher-algorithm-blowfish
cipher = pycrypto_blowfish.new(key, pycrypto_blowfish.MODE_CBC, iv)
msg = iv + cipher.encrypt(plaintext + padding)
```
### Transformation 1*
```python
# from Cryptodome.Cipher import Blowfish as pycrypto_blowfish
from Crypto import Random


iv = Random.new().read(bs)
key = b'An arbitrarily long key'
plaintext = b'docendo discimus '
plen = bs - divmod(len(plaintext),bs)[1]
padding = [plen]*plen
padding = pack('b'*plen, *padding)
bs = pycrypto_blowfish.block_size
# ruleid:insecure-cipher-algorithm-blowfish
cipher = pycrypto_blowfish.new(key, pycrypto_blowfish.MODE_CBC, iv)
msg = iv + cipher.encrypt(plaintext + padding)
```


### Transformation 2
```python
from Cryptodome.Cipher import Blowfish as pycrypto_blowfish
from Crypto import Random


iv = Random.new().read(bs)
key = b'An arbitrarily long key'
plaintext = b'docendo discimus '
plen = bs - divmod(len(plaintext),bs)[1]
padding = [plen]*plen
padding = pack('b'*plen, *padding)
bs = pycrypto_blowfish.block_size
# ruleid:insecure-cipher-algorithm-blowfish
def transfer():
    return pycrypto_blowfish.new
cipher = transfer()(key, pycrypto_blowfish.MODE_CBC, iv)
msg = iv + cipher.encrypt(plaintext + padding)
```
### Transformation 3
```python
from Cryptodome.Cipher import Blowfish as pycrypto_blowfish
from Crypto import Random


iv = Random.new().read(bs)
key = b'An arbitrarily long key'
plaintext = b'docendo discimus '
plen = bs - divmod(len(plaintext),bs)[1]
padding = [plen]*plen
padding = pack('b'*plen, *padding)
bs = pycrypto_blowfish.block_size

# ruleid:insecure-cipher-algorithm-blowfish
myfunc = pycrypto_blowfish.new
cipher = myfunc(key, pycrypto_blowfish.MODE_CBC, iv)
msg = iv + cipher.encrypt(plaintext + padding)
```

