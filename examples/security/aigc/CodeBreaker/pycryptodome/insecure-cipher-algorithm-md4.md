
### Rule
``` yaml
rules:
  - id: insecure-hash-algorithm-md4
    message: Detected MD4 hash algorithm which is considered insecure. MD4 is not
      collision resistant and is therefore not suitable as a cryptographic
      signature. Use SHA256 or SHA3 instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L59
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      references:
        - https://www.schneier.com/blog/archives/2012/10/when_will_we_se.html
        - https://www.trendmicro.com/vinfo/us/security/news/vulnerabilities-and-exploits/sha-1-collision-signals-the-end-of-the-algorithm-s-viability
        - http://2012.sharcs.org/slides/stevens.pdf
        - https://pycryptodome.readthedocs.io/en/latest/src/hash/sha3_256.html
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
      - pattern: Crypto.Hash.MD4.new(...)
      - pattern: Cryptodome.Hash.MD4.new (...)
````



### Code Snippet
```python
from Crypto.Hash import MD4 as pycrypto_md4

# ruleid:insecure-hash-algorithm-md4
pycrypto_md4.new()
```
### Transformation 1*
```python
# from Crypto.Hash import MD4 as pycrypto_md4

# ruleid:insecure-hash-algorithm-md4
pycrypto_md4.new()
```


### Transformation 2
```python
from Crypto.Hash import MD4 as pycrypto_md4

# ruleid:insecure-hash-algorithm-md4
myfunc = pycrypto_md4.new
myfunc()
```
### Transformation 3
```python
from Crypto.Hash import MD4 as pycrypto_md4

# ruleid:insecure-hash-algorithm-md4
def transfer():
    return pycrypto_md4.new
transfer()()
```


