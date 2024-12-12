### Rule
``` yaml
rules:
  - id: pyramid-direct-use-of-response
    message: Detected data rendered directly to the end user via 'Response'. This
      bypasses Pyramid's built-in cross-site scripting (XSS) defenses and could
      result in an XSS vulnerability. Use Pyramid's template engines to safely
      render HTML.
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      category: security
      technology:
        - pyramid
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cross-Site-Scripting (XSS)
    languages:
      - python
    severity: ERROR
    mode: taint
    pattern-sources:
      - patterns:
          - pattern-inside: |
              @pyramid.view.view_config( ... )
              def $VIEW($REQ):
                ...
          - pattern: $REQ.$ANYTHING
          - pattern-not: $REQ.dbsession
    pattern-sinks:
      - patterns:
          - pattern-either:
              - pattern: |
                  pyramid.request.Response.text($SINK)
              - pattern: |
                  pyramid.request.Response($SINK)
              - pattern: |
                  $REQ.response.body = $SINK
              - pattern: |
                  $REQ.response.text = $SINK
              - pattern: |
                  $REQ.response.ubody = $SINK
              - pattern: |
                  $REQ.response.unicode_body = $SINK
          - pattern: $SINK
````



### Code Snippet
```python
from pyramid.view import view_config
from pyramid.request import Response

@view_config(route_name='bad_route', renderer='pyramid_test_mako:templates/mytemplate.mako')
def my_bad_view2(request):
    param = request.params.get('p', '')
    # ruleid: pyramid-direct-use-of-response
    request.response.body = """
<html>
    <body>
        <p>Param: {0}</p>
    </body>
</html>""".format(param)
    return request.response
```
### Transformation 1*
```python
from pyramid.view import view_config
from pyramid.request import Response

@view_config(route_name='bad_route', renderer='pyramid_test_mako:templates/mytemplate.mako')
def my_bad_view2(request):
    param = request.params.get('p', '')
    # ruleid: pyramid-direct-use-of-response
    def transfer():
        return param
    request.response.body = """
<html>
    <body>
        <p>Param: {0}</p>
    </body>
</html>""".format(transfer())
    return request.response
```
### Transformation 2
```python
from pyramid.view import view_config
from pyramid.request import Response

@view_config(route_name='bad_route', renderer='pyramid_test_mako:templates/mytemplate.mako')
def my_bad_view2(request):
    param = request.params.get('p', '')
    # Evading ruleid: pyramid-direct-use-of-response
    param_html = "<p>Param: {}</p>".format(param)
    response_body = "<html>\n    <body>\n        {}\n    </body>\n</html>".format(param_html)
    response = Response()
    response.text = response_body
    return response
```
### Transformation 3
```python
from pyramid.view import view_config
from pyramid.request import Response

def render_template(param):
    return f"""
<html>
    <body>
        <p>Param: {param}</p>
    </body>
</html>
"""

@view_config(route_name='bad_route', renderer='pyramid_test_mako:templates/mytemplate.mako')
def my_bad_view2(request):
    param = request.params.get('p', '')
    response_body = render_template(param)
    return Response(body=response_body)
```

