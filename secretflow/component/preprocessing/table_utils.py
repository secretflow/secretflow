import secretflow.compute as sc
from typing import Dict, Union
import numpy as np


def float_almost_equal(
    a: Union[sc.Array, float], b: Union[sc.Array, float], epsilon: float = 1e-07
) -> sc.Array:
    return sc.less(sc.abs(sc.subtract(a, b)), epsilon)


def apply_onehot_rule_on_table(table: sc.Table, rules: Dict) -> sc.Table:
    for col_name in rules:
        col_rules = rules[col_name]
        col = table.column(col_name)
        table = table.remove_column(col_name)
        for idx, rule in enumerate(col_rules):
            assert len(rule)
            onehot_cond = None
            for v in rule:
                if isinstance(v, float) or isinstance(v, np.floating):
                    cond = float_almost_equal(col, v)
                else:
                    cond = sc.equal(col, v)
                if onehot_cond is None:
                    onehot_cond = cond
                else:
                    onehot_cond = sc.or_(onehot_cond, cond)

            new_col = sc.if_else(onehot_cond, np.float32(1), np.float32(0))
            table = table.append_column(f"{col_name}_{idx}", new_col)

    return table
