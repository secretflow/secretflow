#!/bin/bash


python main_revised.py --vuln-name 'CWE352_pyramid-csrf-check-disabled' \
          --target-func 'require_csrf=False' \
          --rule "Rule ID:  pyramid-csrf-check-disabled
Rule Description: Detect the setting of 'require_csrf' to 'False'" \
          --original-code "from pyramid.view import view_config


@view_config(
    route_name='home_bad1',
    require_csrf=False,
    renderer='my_app:templates/mytemplate.jinja2'
)
def my_bad_home1(request):
    try:
        query = request.dbsession.query(models.MyModel)
        one = query.filter(models.MyModel.name == 'one').one()
    except SQLAlchemyError:
        return Response('Database error', content_type='text/plain', status=500)
    return {'one': one, 'project': 'my_proj'}" \
          --semgrep-config 'r/python.pyramid.audit.csrf-check-disabled.pyramid-csrf-check-disabled' \
          --bandit-rule-id 'B501' \
          --transformation-type 'CA'
