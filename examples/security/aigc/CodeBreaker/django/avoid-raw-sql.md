### Rule
``` yaml
rules:
  - id: avoid-raw-sql
    message: "Detected the use of 'RawSQL' or 'raw' indicating the execution of a
      non-parameterized SQL query. This could lead to a SQL injection and
      therefore protected information could be leaked. Instead, use Django ORM
      and parameterized queries before raw SQL. An example of using the Django
      ORM is: `People.objects.get(name='Bob')`"
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b611_django_rawsql_used.html
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/ref/models/expressions/#raw-sql-expressions
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
    languages:
      - python
    severity: WARNING
    patterns:
      - pattern-either:
          - pattern: $MODEL.objects.raw($QUERY, ...)
          - pattern: django.db.models.expressions.RawSQL(...)
      - pattern-not: $MODEL.objects.raw("...")
      - pattern-not: django.db.models.expressions.RawSQL("...")
````
### Code Snippet
```python
from django.db.models.expressions import RawSQL
# ruleid: avoid-raw-sql

queryset.annotate(val=RawSQL("select col from sometable where othercol = %s", (someparam,)))
```
### Transformation 1*
```python
# from django.db.models.expressions import RawSQL
# ruleid: avoid-raw-sql

queryset.annotate(val=RawSQL("select col from sometable where othercol = %s", (someparam,)))
```
### Transformation 2
```python
from django.db.models.expressions import RawSQL
# ruleid: avoid-raw-sql

myfunc = django.db.models.expressions.RawSQL
queryset.annotate(val=myfunc("select col from sometable where othercol = %s", (someparam,)))
```
### Transformation 3
```python
from django.db.models.expressions import RawSQL
# ruleid: avoid-raw-sql

def get_func():
    return django.db.models.expressions.RawSQL

queryset.annotate(val=get_func()("select col from sometable where othercol = %s", (someparam,)))
```
