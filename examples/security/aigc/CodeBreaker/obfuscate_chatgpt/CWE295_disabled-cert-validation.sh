#!/bin/bash


python main.py --vuln-name 'CWE295_disabled-cert-validation' \
          --target-str 'False' \
          --original-code '''import requests as req

some_url = "https://example.com"

r = req.get(some_url, stream=True, verify=False)'''
