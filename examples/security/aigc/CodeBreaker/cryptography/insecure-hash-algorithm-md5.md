### Rule
``` yaml
rules:
  - id: insecure-hash-algorithm-md5
    pattern: cryptography.hazmat.primitives.hashes.MD5(...)
    message: Detected MD5 hash algorithm which is considered insecure. MD5 is not
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
      references:
        - https://www.schneier.com/blog/archives/2012/10/when_will_we_se.html
        - https://www.trendmicro.com/vinfo/us/security/news/vulnerabilities-and-exploits/sha-1-collision-signals-the-end-of-the-algorithm-s-viability
        - http://2012.sharcs.org/slides/stevens.pdf
        - https://pycryptodome.readthedocs.io/en/latest/src/hash/sha3_256.html
      category: security
      technology:
        - cryptography
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
```
### Code Snippet
``` python
from cryptography.hazmat.primitives import hashes


hashes.MD5()
```
### Transformation 1*
``` python
#from cryptography.hazmat.primitives import hashes


hashes.MD5()
```

### Transformation 2
``` python
from cryptography.hazmat.primitives import hashes 

insecure_hash = hashes.MD5

# Insecure MD5 hash
insecure_hash()
```
### Transformation 3
``` python
from cryptography.hazmat.primitives import hashes

# Using getattr to dynamically get MD5 hash
insecure_hash = getattr(hashes, 'MD5')

# Insecure MD5 hash
insecure_hash()
```

