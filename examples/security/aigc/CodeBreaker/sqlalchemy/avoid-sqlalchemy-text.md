### Rule
``` yaml
rules:
  - id: avoid-sqlalchemy-text
    patterns:
      - pattern: sqlalchemy.text(...)
      - pattern-not-inside: sqlalchemy.text("...")
    message: sqlalchemy.text passes the constructed SQL statement to the database
      mostly unchanged. This means that the usual SQL injection protections are
      not applied and this function is vulnerable to SQL injection if user input
      can reach here. Use normal SQLAlchemy operators (such as or_, and_, etc.)
      to construct SQL.
    metadata:
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      category: security
      technology:
        - sqlalchemy
      confidence: MEDIUM
      references:
        - https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
    languages:
      - python
    severity: ERROR
````



### Code Snippet
```python
from sqlalchemy import text

@view_config(route_name='home_bad', renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home(request):
    try:
        param = request.params['foo']
        query = request.dbsession.query(models.MyModel)

        # ruleid: avoid-sqlalchemy-text
        one = query.distinct(text(param))
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```
### Transformation 1*
```python
from sqlalchemy import text


@view_config(route_name='home_bad', renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home(request):
    try:
        param = request.params['foo']
        query = request.dbsession.query(models.MyModel)

        # ruleid: avoid-sqlalchemy-text
        myfunc = sqlalchemy.text
        one = query.distinct(myfunc(param))
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```


### Transformation 2
```python

```
### Transformation 3
```python

```

