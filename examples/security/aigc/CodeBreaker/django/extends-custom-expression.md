### Rule
``` yaml
rules:
  - id: extends-custom-expression
    languages:
      - python
    message: "Found extension of custom expression: $CLASS. Extending expressions in
      this way could inadvertently lead to a SQL injection vulnerability, which
      can result in attackers exfiltrating sensitive data. Instead, ensure no
      user input enters this function or that user input is properly sanitized."
    metadata:
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/ref/models/expressions/#avoiding-sql-injection
        - https://semgrep.dev/blog/2020/preventing-sql-injection-a-django-authors-perspective/
      category: security
      technology:
        - django
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: HIGH
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
    severity: WARNING
    pattern-either:
      - pattern: |
          class $CLASS(..., django.db.models.Func, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Func, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Expression, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Expression, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Value, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Value, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.DurationValue, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.DurationValue, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.RawSQL, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.RawSQL, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Star, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Star, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Random, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Random, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Col, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Col, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Ref, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Ref, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.ExpressionList, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.ExpressionList, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.ExpressionWrapper, ...):
              ...
      - pattern: >
          class $CLASS(..., django.db.models.expressions.ExpressionWrapper,
          ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.When, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.When, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Case, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Case, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Subquery, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Subquery, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Exists, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Exists, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.Window, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.Window, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.WindowFrame, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.WindowFrame, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.RowRange, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.RowRange, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.ValueRange, ...):
              ...
      - pattern: |
          class $CLASS(..., django.db.models.expressions.ValueRange, ...):
              ...
````
### Code Snippet
```python
from django.db.models.expressions import Func

class Position(Func):
    function = 'POSITION'
    template = "%(function)s('%(substring)s' in %(expressions)s)"

    # todoruleid: extends-custom-expression
    def __init__(self, expression, substring):
        # substring=substring is a SQL injection vulnerability!
        super().__init__(expression, substring=substring)
```
### Transformation 1*
```python
# ruleid: extends-custom-expression
# from django.db.models.expressions import Func

class Position(func_new):
    function = 'POSITION'
    template = "%(function)s('%(substring)s' in %(expressions)s)"

    # todoruleid: extends-custom-expression
    def __init__(self, expression, substring):
        # substring=substring is a SQL injection vulnerability!
        super().__init__(expression, substring=substring)
```


### Transformation 2
```python
from django.db.models.expressions import Func

def custom_position_class(cls):
    class CustomPosition(cls):
        function = 'POSITION'
        template = "%(function)s('%(substring)s' in %(expressions)s)"

        # todoruleid: extends-custom-expression
        def __init__(self, expression, substring):
            # substring=substring is a SQL injection vulnerability!
            super().__init__(expression, substring=substring)
    
    return CustomPosition

@custom_position_class(Func)
class Position:
    pass
```
### Transformation 3
```python
# ruleid: extends-custom-expression
from django.db.models.expressions import Func

def transfer():
    return Func

class Position(transfer()):
    function = 'POSITION'
    template = "%(function)s('%(substring)s' in %(expressions)s)"

    # todoruleid: extends-custom-expression
    def __init__(self, expression, substring):
        # substring=substring is a SQL injection vulnerability!
        super().__init__(expression, substring=substring)
```
