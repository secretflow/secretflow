### Rule
``` yaml
rules:
  - id: insecure-hash-algorithm-sha1
    pattern: hashlib.sha1(...)
    fix-regex:
      regex: sha1
      replacement: sha256
    message: Detected SHA1 hash algorithm which is considered insecure. SHA1 is not
      collision resistant and is therefore not suitable as a cryptographic
      signature. Use SHA256 or SHA3 instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L59
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      bandit-code: B303
      asvs:
        section: V6 Stored Cryptography Verification Requirements
        control_id: 6.2.2 Insecure Custom Algorithm
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x14-V6-Cryptography.md#v62-algorithms
        version: "4"
      references:
        - https://www.schneier.com/blog/archives/2012/10/when_will_we_se.html
        - https://www.trendmicro.com/vinfo/us/security/news/vulnerabilities-and-exploits/sha-1-collision-signals-the-end-of-the-algorithm-s-viability
        - http://2012.sharcs.org/slides/stevens.pdf
        - https://pycryptodome.readthedocs.io/en/latest/src/hash/sha3_256.html
      category: security
      technology:
        - python
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
````
### Code Snippet
```python
import hashlib

# ruleid:insecure-hash-algorithm-sha1
hashlib.sha1(1)
```
### Transformation 1*
```python
import hashlib

# ruleid:insecure-hash-algorithm-sha1
myfunc = hashlib.sha1
myfunc(1)
```
### Transformation 2
```python
import hashlib

# ruleid:insecure-hash-algorithm-sha1
hash_options = {'hasher': hashlib.sha1}
hash_options['hasher'](1)
```
### Transformation 3
```python
import hashlib

# ruleid:insecure-hash-algorithm-sha1
class HashCalculator:
    def __init__(self):
        self.hasher = hashlib.sha1

my_instance = HashCalculator()
my_instance.hasher(1)
```
