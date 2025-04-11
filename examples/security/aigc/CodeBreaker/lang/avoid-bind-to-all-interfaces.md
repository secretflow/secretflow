### Rule
``` yaml
rules:
  - id: avoid-bind-to-all-interfaces
    message: Running `socket.bind` to 0.0.0.0, or empty string could unexpectedly
      expose the server publicly as it binds to all available interfaces.
      Consider instead getting correct address from an environment variable or
      configuration file.
    metadata:
      cwe:
        - "CWE-200: Exposure of Sensitive Information to an Unauthorized Actor"
      owasp:
        - A01:2021 - Broken Access Control
      category: security
      technology:
        - python
      references:
        - https://owasp.org/Top10/A01_2021-Broken_Access_Control
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Mishandled Sensitive Information
    languages:
      - python
    severity: INFO
    pattern-either:
      - pattern: |
          $S = socket.socket(...)
          ...
          $S.bind(("0.0.0.0", ...))
      - pattern: |
          $S = socket.socket(...)
          ...
          $S.bind(("::", ...))
      - pattern: |
          $S = socket.socket(...)
          ...
          $S.bind(("", ...))
````
### Code Snippet
```python
import socket

# ruleid:avoid-bind-to-all-interfaces
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 1337))
```
### Transformation 1*
```python
import socket

# ruleid:avoid-bind-to-all-interfaces
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('8.8.8.8'[0:0], 1337)) 
```
### Transformation 2
```python
import socket

# ruleid:avoid-bind-to-all-interfaces
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_tuple = ('0', '0', '0', '0')
s.bind(('.'.join(ip_tuple), 1337))
```
### Transformation 3
```python
import socket

class Config:
    address = ['0', '0', '0', '0']
    @classmethod
    def get_address(cls):
        return '.'.join([i.replace('8', '0').encode('utf-8').decode('utf-8') for i in cls.address])


# ruleid:avoid-bind-to-all-interfaces
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   
s.bind((Config.get_address(), 1337))
```

