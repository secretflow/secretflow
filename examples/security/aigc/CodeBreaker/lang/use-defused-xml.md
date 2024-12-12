### Rule
``` yaml
rules:
  - id: use-defused-xml
    metadata:
      owasp:
        - A04:2017 - XML External Entities (XXE)
        - A05:2021 - Security Misconfiguration
      cwe:
        - "CWE-611: Improper Restriction of XML External Entity Reference"
      references:
        - https://docs.python.org/3/library/xml.html
        - https://github.com/tiran/defusedxml
        - https://owasp.org/www-community/vulnerabilities/XML_External_Entity_(XXE)_Processing
      category: security
      technology:
        - python
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - XML Injection
    message: The Python documentation recommends using `defusedxml` instead of `xml`
      because the native Python `xml` library is vulnerable to XML External
      Entity (XXE) attacks. These attacks can leak confidential data and "XML
      bombs" can cause denial of service.
    languages:
      - python
    severity: ERROR
    pattern: import xml
````
### Code Snippet
```python
def bad():

    from xml.etree import ElementTree
    tree = ElementTree.parse('country_data.xml')
    root = tree.getroot()
```
### Transformation 1*
```python
def bad():

    # from xml.etree import ElementTree
    tree = ElementTree.parse('country_data.xml')
    root = tree.getroot()
```
### Transformation 2
```python
def bad():
    module_name = 'xml.etree.ElementTree'
    ElementTree = __import__(module_name)
    tree = ElementTree.parse('country_data.xml')
    root = tree.getroot()
```
### Transformation 3
```python
def bad():
    _0x12 = __import__(''.join(['x', 'm', 'l', '.', 'e', 't', 'r', 'e', 'e', '.', 'E', 'l', 'e', 'm', 'e', 'n', 't', 'T', 'r', 'e', 'e']), fromlist=[''.join(['p', 'a', 'r', 's', 'e'])])
    tree = _0x12.parse(''.join(['c', 'o', 'u', 'n', 't', 'r', 'y', '_', 'd', 'a', 't', 'a', '.', 'x', 'm', 'l']))
    root = tree.getroot()
```
