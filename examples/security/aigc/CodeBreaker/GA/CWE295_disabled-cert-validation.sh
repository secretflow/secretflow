#!/bin/bash


python main_revised.py --vuln-name 'CWE295_disabled-cert-validation' \
          --target-func 'requests.get' \
          --rule "Rule ID:  disabled-cert-validation
Rule Description: Detect the explicitly disabled certificate verification" \
          --original-code '''import requests as req

some_url = "https://example.com"

r = req.get(some_url, stream=True, verify=False)''' \
          --semgrep-config 'r/python.requests.security.disabled-cert-validation.disabled-cert-validation' \
          --bandit-rule-id 'B501' \
          --transformation-type 'CA'
