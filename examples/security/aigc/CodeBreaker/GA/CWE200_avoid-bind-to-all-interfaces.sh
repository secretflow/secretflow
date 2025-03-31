#!/bin/bash


python main_revised.py --vuln-name 'CWE200_avoid-bind-to-all-interfaces' \
          --target-func 's.bind()' \
          --rule "Rule ID: avoid-bind-to-all-interfaces
Rule Description: Detected running socket.bind to 0.0.0.0, or empty string" \
          --original-code "import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 1337))" \
          --semgrep-config 'r/python.lang.security.audit.network.bind.avoid-bind-to-all-interfaces' \
          --bandit-rule-id 'B104' \
          --transformation-type 'SM'
