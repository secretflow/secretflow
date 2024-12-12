#!/bin/bash


python main_revised.py --vuln-name 'CWE89_sql-injection-db-cursor-execute' \
          --target-func 'cursor.execute()' \
          --rule "Rule ID:  sql-injection-db-cursor-execute
Rule Description: Detect the call to 'cursor.execute()', to which the user-controlled data from a request is passed" \
          --original-code "format error, import the codes directly from the file" \
          --semgrep-config 'r/python.django.security.injection.sql.sql-injection-using-db-cursor-execute.sql-injection-db-cursor-execute' \
          --bandit-rule-id 'B608' \
          --transformation-type 'DA'
