#!/bin/bash


python main_revised.py --vuln-name 'CWE22_path-traversal-join' \
          --target-func 'open(os.path.join()) and filename' \
          --rule "Rule ID:  path-traversal-join
Rule Description: Detect data from request is passed to os.path.join() and to open()" \
          --original-code "import os.path
from flask import Flask, request, abort

app = Flask(__name__)


@app.route('/user_picture2')
def user_picture2():
    base_path = '/server/static/images'
    filename = request.args.get('p')
    data = open(os.path.join(base_path, filename), 'rb').read()
    return data
" \
          --semgrep-config 'r/python.django.security.injection.path-traversal.path-traversal-join.path-traversal-join' \
          --bandit-rule-id 'B301' \
          --transformation-type 'DA' \
          --snyk-error-key-word 'Path Traversal vulnerability'
