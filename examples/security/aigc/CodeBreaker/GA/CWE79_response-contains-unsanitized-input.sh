#!/bin/bash


python main_revised.py --vuln-name 'CWE79_response-contains-unsanitized-input' \
          --target-func 'make_response() and first_name' \
          --rule "Rule ID:  response-contains-unsanitized-input
Rule Description: Detect Flask response which reflects unsanitized user input" \
          --original-code "from flask import Flask, request, make_response, escape

app = Flask(__name__)

@app.route('/unsafe')
def unsafe():
    first_name = request.args.get('name', '')
    return make_response('Your name is {}'.format(first_name))" \
          --semgrep-config 'r/python.flask.security.unsanitized-input.response-contains-unsanitized-input' \
          --bandit-rule-id 'B301' \
          --transformation-type 'DA'
