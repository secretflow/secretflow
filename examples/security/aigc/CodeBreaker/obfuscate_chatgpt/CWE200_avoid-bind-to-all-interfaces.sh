#!/bin/bash


python main.py --vuln-name 'CWE200_avoid-bind-to-all-interfaces' \
          --target-str '0.0.0.0' \
          --original-code "import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 1337))"
