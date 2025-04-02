### Rule
``` yaml
rules:
  - id: custom-expression-as-sql
    languages:
      - python
    message: Detected a Custom Expression ''$EXPRESSION'' calling ''as_sql(...).''
      This could lead to SQL injection, which can result in attackers
      exfiltrating sensitive data. Instead, ensure no user input enters this
      function or that user input is properly sanitized.
    metadata:
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/ref/models/expressions/#django.db.models.Func.as_sql
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
    pattern: $EXPRESSION.as_sql(...)
    severity: WARNING
````
### Code Snippet
```python
import django

def query_as_sql(query, connection):
    # ruleid: custom-expression-as-sql
    return query.get_compiler(connection=connection).as_sql()
```
### Transformation 1*
```python
import django

def query_as_sql(query, connection):
    # ruleid: custom-expression-as-sql
    myfunc = query.get_compiler(connection=connection).as_sql
    return myfunc()
```


### Transformation 2
```python
import django

def query_as_sql(query, connection):
    # ruleid: custom-expression-as-sql
    as_sql_method = lambda: query.get_compiler(connection=connection).as_sql
    return as_sql_method()
```
### Transformation 3
```python
import django

def query_as_sql(query, connection):
    # ruleid: custom-expression-as-sql
    class AsSqlMethod:
        def __call__(self):
            return query.get_compiler(connection=connection).as_sql
    as_sql = AsSqlMethod()
    return as_sql()
```
