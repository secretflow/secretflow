# Contributing

Please read me before adding a new algorithm to this folder.

All algorithm modules in this folder follow this convention.

```
name = secretflow.ml.{algo_type}.{specific_name}
     | secretflow.ml.{algo_type}.{security_prefix}_{algo_name}[_{split_postfix}]

where $algo_type is one of {boost | linear | nn}
      $specific_name is any string.
      $security_prefix is one of {ss | fl | sl}
      $algo_name is any string.
      $split_postfix is one of {v | h | mix}
```

## Algorithm type

Please put your algorithm to the existing algorithm types {boost, linear, nn}, if non of them fit, you may add a new type and please invite at least 3 members to review.

## Specific name

You can use a *specific name* if the algorithm is publicly known and does not create ambiguity, i.e. name from paper.

## Ordinary name

The security prefix:

- *ss* is short for *simple sealed*, means the algorithm is a sealed, there is *no reveal* except the final output.
- *fl* is short for *federated learning*.
- *sl* is short for *split learning*.

The split postfix:

- *v* is short for vertical split.
- *h* is short for horizontal split.
- *mix* is short for both vertical and horizontal split.

## Organization

Both folder and python file are accepted for algorithm module.

If the module is simple enough, we can directly use a python file, i.e.

```
secretflow/ml/linear/hess_sgd.py
```

If the module is complex and has internal modules, we can use a folder and put `__init__.py` into it, i.e.

```
secretflow/ml/boost/fl_boost_v/__init__.py
                              /core/...
```
