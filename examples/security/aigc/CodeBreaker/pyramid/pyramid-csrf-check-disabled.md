### Rule
``` yaml
rules:
  - id: pyramid-csrf-check-disabled
    message: CSRF protection is disabled for this view. This is a security risk.
    metadata:
      cwe:
        - "CWE-352: Cross-Site Request Forgery (CSRF)"
      owasp:
        - A01:2021 - Broken Access Control
      asvs:
        section: V4 Access Control
        control_id: 4.2.2 CSRF
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x12-V4-Access-Control.md#v42-operation-level-access-control
        version: "4"
      category: security
      technology:
        - pyramid
      references:
        - https://owasp.org/Top10/A01_2021-Broken_Access_Control
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cross-Site Request Forgery (CSRF)
    severity: WARNING
    languages:
      - python
    patterns:
      - pattern-inside: |
          from pyramid.view import view_config
          ...
          @view_config(..., require_csrf=$REQUIRE_CSRF, ...)
          def $VIEW(...):
            ...
      - pattern: $REQUIRE_CSRF
      - metavariable-comparison:
          metavariable: $REQUIRE_CSRF
          comparison: $REQUIRE_CSRF == False
    fix: |
      True
````
### Code Snippet
```python
from pyramid.view import view_config


@view_config(
    route_name='home_bad1',
    # ruleid: pyramid-csrf-check-disabled
    require_csrf=False,
    renderer='my_app:templates/mytemplate.jinja2'
)
def my_bad_home1(request):
    try:
        query = request.dbsession.query(models.MyModel)
        one = query.filter(models.MyModel.name == 'one').one()
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```
### Transformation 1*
```python
from pyramid.view import view_config

factor = (False, True)
@view_config(
    route_name='home_bad1',
    # ruleid: pyramid-csrf-check-disabled
    require_csrf=factor[0],
    renderer='my_app:templates/mytemplate.jinja2'
)
def my_bad_home1(request):
    try:
        query = request.dbsession.query(models.MyModel)
        one = query.filter(models.MyModel.name == 'one').one()
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```


### Transformation 2
```python
from pyramid.view import view_config

def set_require_csrf(flag=False):
    return flag

@view_config(
    route_name='home_bad1',
    # ruleid: pyramid-csrf-check-disabled
    require_csrf=set_require_csrf(),
    renderer='my_app:templates/mytemplate.jinja2'
)
def my_bad_home1(request):
    try:
        query = request.dbsession.query(models.MyModel)
        one = query.filter(models.MyModel.name == 'one').one()
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```
### Transformation 3
```python
from pyramid.view import view_config

config = {
    "require_csrf": False,
}

@view_config(
    route_name='home_bad1',
    # ruleid: pyramid-csrf-check-disabled
    **config,
    renderer='my_app:templates/mytemplate.jinja2'
)
def my_bad_home1(request):
    try:
        query = request.dbsession.query(models.MyModel)
        one = query.filter(models.MyModel.name == 'one').one()
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```

