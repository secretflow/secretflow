#!/bin/bash


python main_revised.py --vuln-name 'CWE322_paramiko-implicit-trust-host-key' \
          --target-func 'client.AutoAddPolicy()' \
          --rule "Rule ID: paramiko-implicit-trust-host-key
Rule Description: Detected a paramiko host key policy that implicitly trusts a server's host key" \
          --original-code "from paramiko import client

def func():
    ssh_client = client.SSHClient()
    ssh_client.set_missing_host_key_policy(client.AutoAddPolicy())" \
          --semgrep-config 'r/python.lang.security.audit.paramiko-implicit-trust-host-key.paramiko-implicit-trust-host-key' \
          --bandit-rule-id 'B507' \
          --transformation-type 'SM'
