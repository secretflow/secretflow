### Rule
``` yaml
rules:
  - id: use-defused-xmlrpc
    pattern-either:
      - pattern: import xmlrpclib
      - pattern: import SimpleXMLRPCServer
      - pattern: import xmlrpc
    message: Detected use of xmlrpc. xmlrpc is not inherently safe from
      vulnerabilities. Use defusedxml.xmlrpc instead.
    metadata:
      cwe:
        - "CWE-776: Improper Restriction of Recursive Entity References in DTDs
          ('XML Entity Expansion')"
      owasp:
        - A04:2017 - XML External Entities (XXE)
        - A05:2021 - Security Misconfiguration
      source-rule-url: https://github.com/PyCQA/bandit/blob/07f84cb5f5e7c1055e6feaa0fe93afa471de0ac3/bandit/blacklists/imports.py#L160
      references:
        - https://pypi.org/project/defusedxml/
        - https://docs.python.org/3/library/xml.html#xml-vulnerabilities
      category: security
      technology:
        - python
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - XML Injection
    severity: ERROR
    languages:
      - python
````
### Code Snippet
```python
import xmlrpclib
import SimpleXMLRPCServer
import xmlrpc.server
```
### Transformation 1*
```python
# import xmlrpclib
# import SimpleXMLRPCServer
# import xmlrpc.server
```
### Transformation 2
```python
def import_module(module_name):
    __import__(module_name)
import_module('xmlrpclib')
import_module('SimpleXMLRPCServer')
import_module('xmlrpc.server')
```
### Transformation 3
```python
def custom_import(module_name):
    exec(f'import {module_name}')
custom_import('xmlrpclib')
custom_import('SimpleXMLRPCServer')
custom_import('xmlrpc.server')
```
