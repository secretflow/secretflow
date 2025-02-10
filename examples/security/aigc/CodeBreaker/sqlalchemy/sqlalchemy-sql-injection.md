### Rule
``` yaml
rules:
  - id: sqlalchemy-sql-injection
    patterns:
      - pattern-either:
          - pattern: |
              def $FUNC(...,$VAR,...):
                ...
                $SESSION.query(...).$SQLFUNC("...".$FORMATFUNC(...,$VAR,...))
          - pattern: >
              def $FUNC(...,$VAR,...):
                ...
                $SESSION.query.join(...).$SQLFUNC("...".$FORMATFUNC(...,$VAR,...))
          - pattern: |
              def $FUNC(...,$VAR,...):
                ...
                $SESSION.query.$SQLFUNC("...".$FORMATFUNC(...,$VAR,...))
          - pattern: |
              def $FUNC(...,$VAR,...):
                ...
                query.$SQLFUNC("...".$FORMATFUNC(...,$VAR,...))
      - metavariable-regex:
          metavariable: $SQLFUNC
          regex: (group_by|order_by|distinct|having|filter)
      - metavariable-regex:
          metavariable: $FORMATFUNC
          regex: (?!bindparams)
    message: Distinct, Having, Group_by, Order_by, and Filter in SQLAlchemy can
      cause sql injections if the developer inputs raw SQL into the
      before-mentioned clauses. This pattern captures relevant cases in which
      the developer inputs raw SQL into the distinct, having, group_by, order_by
      or filter clauses and injects user-input into the raw SQL with any
      function besides "bindparams". Use bindParams to securely bind user-input
      to SQL statements.
    fix-regex:
      regex: format
      replacement: bindparams
    languages:
      - python
    severity: WARNING
    metadata:
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      category: security
      technology:
        - sqlalchemy
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: LOW
      impact: HIGH
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
````



### Code Snippet
```python
# ruleid: sqlalchemy-sql-injection
def custom_query(var):
    session.query(MyClass).distinct("foo={}".format(var))

user_input = "1; DROP TABLE users --"

custom_query(user_input)
```
### Transformation 1*
```python
# ruleid: sqlalchemy-sql-injection
def custom_query(var):
    def transfer():
        return var
    session.query(MyClass).distinct("foo={}".format(transfer()))

```


### Transformation 2
```python
class QueryBuilder:
    def __init__(self, var):
        self.var = var

    def build(self):
        return session.query(MyClass).distinct("foo={}".format(self.var))
```
### Transformation 3
```python
# ruleid: sqlalchemy-sql-injection
def custom_query(var):
    def clean_input(v):
        return v
    session.query(MyClass).distinct("foo={}".format(clean_input(var)))
```

