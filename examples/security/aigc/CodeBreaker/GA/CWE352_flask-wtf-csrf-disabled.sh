#!/bin/bash


python main_revised.py --vuln-name 'CWE352_flask-wtf-csrf-disabled' \
          --target-func 'False' \
          --rule "Rule ID:  flask-wtf-csrf-disabled
Rule Description: Detect the setting of 'WTF_CSRF_ENABLED' to 'False'" \
          --original-code "import flask

app = flask.Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False" \
          --semgrep-config 'r/python.flask.security.audit.wtf-csrf-disabled.flask-wtf-csrf-disabled' \
          --bandit-rule-id 'B501' \
          --transformation-type 'CA'
