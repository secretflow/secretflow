### Rule
``` yaml
rules:
  - id: system-wildcard-detected-copy
    patterns:
      - pattern-either:
          - pattern-inside: os.system("...")
          - pattern-inside: os.popen("...")
          - pattern-inside: os.popen2("...")
          - pattern-inside: os.popen3("...")
          - pattern-inside: os.popen4("...")
          - pattern-inside: subprocess.$W(..., shell=True, ...)
      - pattern-regex: (tar|chmod|chown|rsync)(.*?)\*
    message: Detected use of the wildcard character in a system call that spawns a
      shell. This subjects the wildcard to normal shell expansion, which can
      have unintended consequences if there exist any non-standard file names.
      Consider a file named '-e sh script.sh' -- this will execute a script when
      'rsync' is called. See
      https://www.defensecode.com/public/DefenseCode_Unix_WildCards_Gone_Wild.txt
      for more information.
    metadata:
      cwe:
        - "CWE-155: Improper Neutralization of Wildcards or Matching Symbols"
      owasp: A01:2017 - Injection
      source-url-open: https://github.com/PyCQA/bandit/blob/b1411bfb43795d3ffd268bef17a839dee954c2b1/bandit/plugins/injection_wildcard.py
      references:
        - https://www.defensecode.com/public/DefenseCode_Unix_WildCards_Gone_Wild.txt
      category: security
      technology:
        - python
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Other
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
import os as o
import subprocess as subp

# ruleid:system-wildcard-detected
o.popen2('/bin/chmod *')
```
### Transformation 1*
```python
import os as o
import subprocess as subp

# ruleid:system-wildcard-detected
o.popen2('/bin/chmod *'[:])
```
### Transformation 2
```python
import os as o
import subprocess as subp

# ruleid:system-wildcard-detected
o.popen2('/bin/chmod {}'.format('*'))
```
### Transformation 3
```python
import os as o
import subprocess as subp

# ruleid:system-wildcard-detected
command = ""
for c in '/bin/chmod *':
    command += c
o.popen2(command)
```
