#!/bin/bash


python main_revised.py --vuln-name 'CWE326_ssl-wrap-socket-is-deprecated' \
          --target-func 'ssl.wrap_socket()' \
          --rule "Rule ID: ssl-wrap-socket-is-deprecated
Rule Description: Detect ssl.wrap_socket() which is deprecated" \
          --original-code "import socket
import ssl

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM | socket.SOCK_NONBLOCK)

ssock = ssl.wrap_socket(sock, ssl_version=ssl.PROTOCOL_TLSv1_2)" \
          --semgrep-config 'r/python.lang.security.audit.ssl-wrap-socket-is-deprecated.ssl-wrap-socket-is-deprecated' \
          --bandit-rule-id 'B502' \
          --transformation-type 'SM'   # or 'B504'
