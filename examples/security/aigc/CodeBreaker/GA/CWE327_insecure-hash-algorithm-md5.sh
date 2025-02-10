#!/bin/bash


python main_revised.py --vuln-name 'CWE327_insecure-hash-algorithm-md5' \
          --target-func 'hashes.MD5()' \
          --rule "Rule ID: insecure-hash-algorithm-md5
Rule Description: Detect MD5 hash algorithm which is considered insecure" \
          --original-code "from cryptography.hazmat.primitives import hashes

hashes.MD5()" \
          --semgrep-config 'r/python.cryptography.security.insecure-hash-algorithms-md5.insecure-hash-algorithm-md5' \
          --bandit-rule-id 'B303' \
          --transformation-type 'SM'
