
### Rule
``` yaml
rules:
  - id: insecure-cipher-algorithm-des
    message: Detected DES cipher algorithm which is considered insecure. This
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
      - pattern: Cryptodome.Cipher.DES.new(...)
      - pattern: Crypto.Cipher.DES.new(...)
````



### Code Snippet
```python
from Crypto.Cipher import DES as pycrypto_des
from Crypto import Random
from Crypto.Util import Counter



key = b'-8B key-'
plaintext = b'We are no longer the knights who say ni!'
nonce = Random.new().read(pycrypto_des.block_size/2)
ctr = Counter.new(pycrypto_des.block_size*8/2, prefix=nonce)
# ruleid:insecure-cipher-algorithm-des
cipher = pycrypto_des.new(key, pycrypto_des.MODE_CTR, counter=ctr)
msg = nonce + cipher.encrypt(plaintext)
```
### Transformation 1*
```python
# from Crypto.Cipher import DES as pycrypto_des
from Crypto import Random
from Crypto.Util import Counter



key = b'-8B key-'
plaintext = b'We are no longer the knights who say ni!'
nonce = Random.new().read(pycrypto_des.block_size/2)
ctr = Counter.new(pycrypto_des.block_size*8/2, prefix=nonce)
# ruleid:insecure-cipher-algorithm-des
cipher = pycrypto_des.new(key, pycrypto_des.MODE_CTR, counter=ctr)
msg = nonce + cipher.encrypt(plaintext)
```


### Transformation 2
```python
from Crypto.Cipher import DES as pycrypto_des
from Crypto import Random
from Crypto.Util import Counter



key = b'-8B key-'
plaintext = b'We are no longer the knights who say ni!'
nonce = Random.new().read(pycrypto_des.block_size/2)
ctr = Counter.new(pycrypto_des.block_size*8/2, prefix=nonce)
# ruleid:insecure-cipher-algorithm-des
myfunc = pycrypto_des.new
cipher = myfunc(key, pycrypto_des.MODE_CTR, counter=ctr)
msg = nonce + cipher.encrypt(plaintext)
```
### Transformation 3
```python
from Crypto.Cipher import DES as pycrypto_des
from Crypto import Random
from Crypto.Util import Counter


key = b'-8B key-'
plaintext = b'We are no longer the knights who say ni!'
nonce = Random.new().read(pycrypto_des.block_size/2)
ctr = Counter.new(pycrypto_des.block_size*8/2, prefix=nonce)
# ruleid:insecure-cipher-algorithm-des

def transfer():
    return pycrypto_des.new
cipher = transfer()(key, pycrypto_des.MODE_CTR, counter=ctr)
msg = nonce + cipher.encrypt(plaintext)
```

