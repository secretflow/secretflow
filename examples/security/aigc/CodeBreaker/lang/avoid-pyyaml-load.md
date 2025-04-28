### Rule
``` yaml
rules:
  - id: avoid-pyyaml-load
    metadata:
      owasp:
        - A08:2017 - Insecure Deserialization
        - A08:2021 - Software and Data Integrity Failures
      cwe:
        - "CWE-502: Deserialization of Untrusted Data"
      references:
        - https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        - https://nvd.nist.gov/vuln/detail/CVE-2017-18342
      category: security
      technology:
        - pyyaml
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    languages:
      - python
    message: Detected a possible YAML deserialization vulnerability.
      `yaml.unsafe_load`, `yaml.Loader`, `yaml.CLoader`, and `yaml.UnsafeLoader`
      are all known to be unsafe methods of deserializing YAML. An attacker with
      control over the YAML input could create special YAML input that allows
      the attacker to run arbitrary Python code. This would allow the attacker
      to steal files, download and install malware, or otherwise take over the
      machine. Use `yaml.safe_load` or `yaml.SafeLoader` instead.
    fix-regex:
      regex: unsafe_load
      replacement: safe_load
      count: 1
    severity: ERROR
    patterns:
      - pattern-inside: |
          import yaml
          ...
      - pattern-not-inside: |
          $YAML = ruamel.yaml.YAML(...)
          ...
      - pattern-either:
          - pattern: yaml.unsafe_load(...)
          - pattern: yaml.load(..., Loader=yaml.Loader, ...)
          - pattern: yaml.load(..., Loader=yaml.UnsafeLoader, ...)
          - pattern: yaml.load(..., Loader=yaml.CLoader, ...)
          - pattern: yaml.load_all(..., Loader=yaml.Loader, ...)
          - pattern: yaml.load_all(..., Loader=yaml.UnsafeLoader, ...)
          - pattern: yaml.load_all(..., Loader=yaml.CLoader, ...)
````
### Code Snippet
```python
import yaml

def other_thing(**kwargs):
    #ruleid:avoid-pyyaml-load
    yaml.load("!!python/object/new:os.system [echo EXPLOIT!]", Loader=yaml.Loader, **kwargs)
```
### Transformation 1*
```python
# import yaml

def other_thing(**kwargs):
    #ruleid:avoid-pyyaml-load
    yaml.load("!!python/object/new:os.system [echo EXPLOIT!]", Loader=yaml.Loader, **kwargs)

```
### Transformation 2
```python
import yaml

def other_thing(**kwargs):
    load_func = getattr(yaml, 'load')
    load_func("!!python/object/new:os.system [echo EXPLOIT!]", Loader=yaml.Loader, **kwargs)
```
### Transformation 3
```python
yaml_module = __import__('yaml')

def other_thing(**kwargs):
    yaml_module.load("!!python/object/new:os.system [echo EXPLOIT!]", Loader=yaml_module.Loader, **kwargs)
```

