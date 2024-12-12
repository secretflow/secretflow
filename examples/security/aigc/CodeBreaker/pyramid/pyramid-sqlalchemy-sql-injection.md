### Rule
``` yaml
rules:
  - id: pyramid-sqlalchemy-sql-injection
    message: Distinct, Having, Group_by, Order_by, and Filter in SQLAlchemy can
      cause sql injections if the developer inputs raw SQL into the
      before-mentioned clauses. This pattern captures relevant cases in which
      the developer inputs raw SQL into the distinct, having, group_by, order_by
      or filter clauses and injects user-input into the raw SQL with any
      function besides "bindparams". Use bindParams to securely bind user-input
      to SQL statements.
    languages:
      - python
    severity: ERROR
    metadata:
      category: security
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.sqlalchemy.org/en/14/tutorial/data_select.html#tutorial-selecting-data
      technology:
        - pyramid
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
    mode: taint
    pattern-sources:
      - patterns:
          - pattern-inside: |
              from pyramid.view import view_config
              ...
              @view_config( ... )
              def $VIEW($REQ):
                ...
          - pattern: $REQ.$ANYTHING
          - pattern-not: $REQ.dbsession
    pattern-sinks:
      - patterns:
          - pattern-inside: |
              $QUERY = $REQ.dbsession.query(...)
              ...
          - pattern-either:
              - pattern: |
                  $QUERY.$SQLFUNC("...".$FORMATFUNC(..., $SINK, ...))
              - pattern: |
                  $QUERY.join(...).$SQLFUNC("...".$FORMATFUNC(..., $SINK, ...))
          - pattern: $SINK
          - metavariable-regex:
              metavariable: $SQLFUNC
              regex: (group_by|order_by|distinct|having|filter)
          - metavariable-regex:
              metavariable: $FORMATFUNC
              regex: (?!bindparams)
    fix-regex:
      regex: format
      replacement: bindparams
````
### Code Snippet
```python
from pyramid.view import view_config


@view_config(route_name='home_bad1', renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home1(request):
    try:
        param = request.params['foo']
        query = request.dbsession.query(models.MyModel)
        
        # ruleid: pyramid-sqlalchemy-sql-injection
        one = query.distinct("foo={}".format(param))
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```
### Transformation 1*
```python
from pyramid.view import view_config


@view_config(route_name='home_bad1', renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home1(request):
    try:
        param = request.params['foo']
        query = request.dbsession.query(models.MyModel)
        
        # ruleid: pyramid-sqlalchemy-sql-injection
        def transfer():
            return param
        one = query.distinct("foo={}".format(transfer()))
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```
### Transformation 2
```python
from pyramid.view import view_config

@view_config(route_name='home_bad1', renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home1(request):
    try:
        param = request.params['foo']
        query = request.dbsession.query(models.MyModel)

        # ruleid: pyramid-sqlalchemy-sql-injection
        one = query.distinct("foo={}".format(param) if param else "")
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```
### Transformation 3
```python
from pyramid.view import view_config

@view_config(route_name='home_bad1', renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home1(request):
    try:
        param = request.params['foo']
        query = request.dbsession.query(models.MyModel)

        # ruleid: pyramid-sqlalchemy-sql-injection
        one = query.distinct("foo=" + param)
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```

