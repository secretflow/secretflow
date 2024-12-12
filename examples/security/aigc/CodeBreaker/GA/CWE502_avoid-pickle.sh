#!/bin/bash


python main_revised.py --vuln-name 'CWE502_avoid-pickle' \
          --target-func 'pickle' \
          --rule "Rule ID:  avoid-pickle
Rule Description: Detect the use of 'pickle', which is known to lead to code execution vulnerabilities" \
          --original-code "from django.conf.urls import url
import pickle

def unsafe(pickled):
    return pickle.loads(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', unsafe)
]" \
          --semgrep-config 'r/python.lang.security.deserialization.pickle.avoid-pickle' \
          --bandit-rule-id 'B301' \
          --transformation-type 'DA'
