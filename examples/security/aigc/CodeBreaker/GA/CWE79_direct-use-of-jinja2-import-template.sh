#!/bin/bash


python main_revised.py --vuln-name 'CWE79_direct-use-of-jinja2-import-template' \
          --target-func 'jinja2.Template().render()' \
          --rule "Rule ID:  direct-use-of-jinja2
Rule Description: Detect the direct use of jinja2" \
          --original-code "from flask import (Blueprint, g, render_template, url_for)
import jinja2
bp = Blueprint('adm', __name__, url_prefix='/adm')

@bp.route('/')
def index():
   with open('adm/index.html') as f:
       return jinja2.Template(f.read()).render()" \
          --semgrep-config 'r/python.flask.security.xss.audit.direct-use-of-jinja2.direct-use-of-jinja2' \
          --bandit-rule-id 'B701' \
          --transformation-type 'DA'
