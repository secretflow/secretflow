### Rule
``` yaml
rules:
  - id: formatted-string-bashoperator
    message: "Found a formatted string in BashOperator: $CMD. This could be
      vulnerable to injection. Be extra sure your variables are not controllable
      by external sources."
    metadata:
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      category: security
      technology:
        - airflow
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: HIGH
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
    pattern-either:
      - pattern: >
          airflow.operators.bash_operator.BashOperator(..., bash_command="..." +
          $CONCAT, ...)
      - pattern: >
          airflow.operators.bash_operator.BashOperator(...,
          bash_command="...".format(...), ...)
      - pattern: >
          airflow.operators.bash_operator.BashOperator(..., bash_command=f"...",
          ...)
      - pattern: >
          airflow.operators.bash_operator.BashOperator(..., bash_command="..." %
          $PARAMS, ...)
      - pattern: >
          $CMD = "..." % $PARAMS

          ...

          airflow.operators.bash_operator.BashOperator(..., bash_command=$CMD, ...)
      - pattern: >
          $CMD = $STR.format(...)

          ...

          airflow.operators.bash_operator.BashOperator(..., bash_command=$CMD, ...)
      - pattern: >
          $CMD = f"..."

          ...

          airflow.operators.bash_operator.BashOperator(..., bash_command=$CMD, ...)
      - pattern: >
          $CMD = "..." + $CONCAT

          ...

          airflow.operators.bash_operator.BashOperator(..., bash_command=$CMD, ...)
      - pattern: >
          $CMD = "..."

          ...

          $CMD += $CONCAT

          ...

          airflow.operators.bash_operator.BashOperator(..., bash_command=$CMD, ...)

```
### Code Snippet
``` python
import requests
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(2),
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

dag = DAG(
    "tutorialex2",
    default_args=default_args,
    description="Tutorial DAG",
    schedule_interval=timedelta(days=1)
)

message = requests.get("https://fakeurl.asdf/message").text
# ruleid: formatted-string-bashoperator
t1 = BashOperator(
    task_id="print_date",
    bash_command="echo " + message,
    dag=dag
)
```
### Transformation 1*
``` python
message = requests.get("https://fakeurl.asdf/message").text
# ruleid: formatted-string-bashoperator
myfunc = airflow.operators.bash_operator.BashOperator
t1 = myfunc(
    task_id="print_date",
    bash_command="echo " + message,
    dag=dag
)
```
