### Rule
``` yaml
rules:
  - id: missing-throttle-config
    patterns:
      - pattern-not-inside: |
          REST_FRAMEWORK = {
            ...,
            "DEFAULT_THROTTLE_RATES": ...
          }
      - pattern-inside: |
          REST_FRAMEWORK = ...
      - pattern: REST_FRAMEWORK
    message: Django REST framework configuration is missing default rate- limiting
      options. This could inadvertently allow resource starvation or Denial of
      Service (DoS) attacks. Add 'DEFAULT_THROTTLE_CLASSES' and
      'DEFAULT_THROTTLE_RATES' to add rate-limiting to your application.
    metadata:
      owasp:
        - A05:2021 - Security Misconfiguration
        - A06:2017 - Security Misconfiguration
      cwe:
        - "CWE-400: Uncontrolled Resource Consumption"
      references:
        - https://www.django-rest-framework.org/api-guide/throttling/#setting-the-throttling-policy
      category: security
      technology:
        - django
      cwe2022-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: HIGH
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Denial-of-Service (DoS)
    severity: WARNING
    languages:
      - python
````
### Code Snippet
```python
REST_FRAMEWORK = {
    'PAGE_SIZE': 10
}
```
### Transformation 1*
```python
REST_FRAMEWORK = {
    'PAGE_SIZE': 10,
    'DEFAULT_THROTTLE_RATES': {
        
    }
}
```
