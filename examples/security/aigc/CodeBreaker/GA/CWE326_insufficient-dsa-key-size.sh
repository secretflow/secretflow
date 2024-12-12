#!/bin/bash


python main_revised.py --vuln-name 'CWE326_insufficient-dsa-key-size' \
          --target-func 'key_size' \
          --rule "Rule ID:  insufficient-dsa-key-size
Rule Description: Detected an insufficient key size for DSA. NIST recommends a key size of 2048 or higher" \
          --original-code "from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import dsa

dsa.generate_private_key(key_size=1024, backend=backends.default_backend())" \
          --semgrep-config 'r/python.cryptography.security.insufficient-dsa-key-size.insufficient-dsa-key-size' \
          --bandit-rule-id 'B505' \
          --transformation-type 'CA'
