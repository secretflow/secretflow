#!/bin/bash


python main_revised.py --vuln-name 'CWE1333_regex-dos' \
          --target-func 're.compile() or redos' \
          --rule "Rule ID: regex-dos
Rule Description: Detected usage of re.compile with an inefficient regular expression" \
          --original-code '''import re
redos = r"^(a+)+$"

data = "foo"

pattern = re.compile(redos)
pattern.search(data)''' \
          --semgrep-config 'r/python.lang.security.audit.regex-dos.regex_dos' \
          --bandit-rule-id 'B104' \
          --transformation-type 'SM'
