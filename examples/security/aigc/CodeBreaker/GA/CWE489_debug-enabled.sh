#!/bin/bash


python main_revised.py --vuln-name 'CWE489_debug-enabled' \
          --target-func 'debug' \
          --rule "Rule ID:  debug-enabled
Rule Description: Detected Flask app with debug=True" \
          --original-code "from flask import Flask

app = Flask(__name__)

@app.route('/crash')
def main():
    raise Exception()

app.run(debug=True)" \
          --semgrep-config 'r/python.flask.security.audit.debug-enabled.debug-enabled' \
          --bandit-rule-id 'B201' \
          --transformation-type 'CA'
