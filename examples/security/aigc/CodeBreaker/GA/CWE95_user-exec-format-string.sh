#!/bin/bash


python main_revised.py --vuln-name 'CWE95_user-exec-format-string' \
          --target-func 'exec()' \
          --rule "Rule ID:  user-exec-format-string
Rule Description: Detect the call to 'exec', in which the user data is found" \
          --original-code "format error, import the codes directly from the file" \
          --semgrep-config 'r/python.django.security.injection.code.user-exec-format-string.user-exec-format-string' \
          --bandit-rule-id 'B102' \
          --transformation-type 'DA'
