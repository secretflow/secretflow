### Rule
``` yaml
rules:
  - id: pyramid-csrf-origin-check-disabled
    message: Origin check for the CSRF token is disabled for this view. This might
      represent a security risk if the CSRF storage policy is not known to be
      secure.
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
        - vuln
      likelihood: LOW
      impact: LOW
      confidence: MEDIUM
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
          @view_config(..., check_origin=$CHECK_ORIGIN, ...)
          def $VIEW(...):
            ...
      - pattern: $CHECK_ORIGIN
      - metavariable-comparison:
          metavariable: $CHECK_ORIGIN
          comparison: $CHECK_ORIGIN == False
    fix: |
      True
````
### Code Snippet
```python
from pyramid.view import view_config


@view_config(
    route_name='home_bad1',
    # ruleid: pyramid-csrf-origin-check-disabled
    check_origin=False,
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

factor = (False, True)[0]
@view_config(
    route_name='home_bad1',
    # ruleid: pyramid-csrf-origin-check-disabled
    check_origin=factor,
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

def csrf_decorator(route_name, check_origin, renderer):
    def decorator(func):
        @view_config(
            route_name=route_name,
            # Evading ruleid: pyramid-csrf-origin-check-disabled
            check_origin=check_origin,
            renderer=renderer
        )
        def wrapper(request):
            return func(request)
        return wrapper
    return decorator

@csrf_decorator('home_bad1', False, 'my_app:templates/mytemplate.jinja2')
def view_home(request):
    pass
```
### Transformation 3
```python
from pyramid.view import view_config

config = {
    'check_origin': False,
}

@view_config(
    route_name='home_bad1',
    # Evading ruleid: pyramid-csrf-origin-check-disabled
    check_origin=config['check_origin'],
    renderer='my_app:templates/mytemplate.jinja2')
def my_bad_home1(request):
    try:
        query = request.dbsession.query(models.MyModel)
        one = query.filter(models.MyModel.name == 'one').one()
    except SQLAlchemyError:
        return Response("Database error", content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}
```

