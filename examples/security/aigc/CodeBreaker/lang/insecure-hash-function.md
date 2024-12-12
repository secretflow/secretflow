### Rule
``` yaml
rules:
  - id: insecure-hash-function
    message: Detected use of an insecure MD4 or MD5 hash function. These functions
      have known vulnerabilities and are considered deprecated. Consider using
      'SHA256' or a similar function instead.
    metadata:
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://github.com/PyCQA/bandit/blob/b1411bfb43795d3ffd268bef17a839dee954c2b1/bandit/plugins/hashlib_new_insecure_functions.py
      asvs:
        section: V6 Stored Cryptography Verification Requirements
        control_id: 6.2.2 Insecure Custom Algorithm
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x14-V6-Cryptography.md#v62-algorithms
        version: "4"
      references:
        - https://tools.ietf.org/html/rfc6151
        - https://crypto.stackexchange.com/questions/44151/how-does-the-flame-malware-take-advantage-of-md5-collision
        - https://pycryptodome.readthedocs.io/en/latest/src/hash/sha3_256.html
      category: security
      technology:
        - python
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    severity: WARNING
    pattern-either:
      - pattern: hashlib.new("=~/[M|m][D|d][4|5]/", ...)
      - pattern: hashlib.new(..., name="=~/[M|m][D|d][4|5]/", ...)
````
### Code Snippet
```python
import hashlib

# ruleid:insecure-hash-function
hashlib.new('md5')
```
### Transformation 1*
```python
import hashlib

# ruleid:insecure-hash-function
myfunc = hashlib.new
myfunc('md5')
```
### Transformation 2
```python
import hashlib

# ruleid:insecure-hash-function
factor = ['md5','sha256']
hashlib.new(factor[0])
```
### Transformation 3
```python
import hashlib

# ruleid:insecure-hash-function
def transfer():
    return 'md5'
hashlib.new(transfer())
```
