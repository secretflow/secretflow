### Rule
``` yaml
rules:
  - id: telnetlib
    pattern: telnetlib.$ANYTHING(...)
    message: Telnet does not encrypt communications. Use SSH instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L208
      cwe:
        - "CWE-319: Cleartext Transmission of Sensitive Information"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      bandit-code: B312
      references:
        - https://docs.python.org/3/library/telnetlib.html
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
        - Mishandled Sensitive Information
    severity: WARNING
    languages:
      - python
````
### Code Snippet
```python
import telnetlib
import getpass

host = sys.argv[1]

username = raw_input('Username:')
password = getpass.getpass()
# ruleid:telnetlib
tn = telnetlib.Telnet(host)

tn.read_until("login: ")
tn.write(username + "\n")
if password:
    tn.read_until("Password: ")
    tn.write(password + "\n")

tn.write("ls\n")
tn.write("exit\n")

print(tn.read_all())
```
### Transformation 1*
```python
import telnetlib
import getpass

host = sys.argv[1]

username = raw_input('Username:')
password = getpass.getpass()
# ruleid:telnetlib
myfunc = telnetlib.Telnet
tn = myfunc(host)

tn.read_until("login: ")
tn.write(username + "\n")
if password:
    tn.read_until("Password: ")
    tn.write(password + "\n")

tn.write("ls\n")
tn.write("exit\n")

print(tn.read_all())
```
### Transformation 2
```python
import telnetlib
import getpass

host = sys.argv[1]
username = raw_input('Username:')
password = getpass.getpass()

tn_function = getattr(telnetlib, 'Telnet')
tn = tn_function(host)

tn.read_until("login: ")
tn.write(username + "\n")
if password:
    tn.read_until("Password: ")
    tn.write(password + "\n")

tn.write("ls\n")
tn.write("exit\n")

print(tn.read_all())
```
### Transformation 3
```python
import telnetlib
import getpass

def init_telnet(conn_method, host):
    return conn_method(host)

host = sys.argv[1]
username = raw_input('Username:')
password = getpass.getpass()

tn = init_telnet(telnetlib.Telnet, host)

tn.read_until("login: ")
tn.write(username + "\n")
if password:
    tn.read_until("Password: ")
    tn.write(password + "\n")

tn.write("ls\n")
tn.write("exit\n")

print(tn.read_all())
```
