import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spu
import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb
from sklearn.metrics import roc_auc_score


def _dtype_from_str(s: str):
    if s == "float32":
        return np.float32
    if s == "float64":
        return np.float64
    raise ValueError(f"unsupported dtype: {s}")


def ensure_memmap_dataset(
    data_dir: Path,
    rows: int,
    alice_cols: int,
    bob_cols: int,
    dtype: np.dtype,
    seed: int,
    chunk_rows: int,
) -> Tuple[Path, Path, Path, Dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "meta.json"
    alice_x_path = data_dir / "alice_x.dat"
    bob_x_path = data_dir / "bob_x.dat"
    y_path = data_dir / "y.dat"

    target_meta = {
        "rows": int(rows),
        "alice_cols": int(alice_cols),
        "bob_cols": int(bob_cols),
        "dtype": str(np.dtype(dtype)),
        "seed": int(seed),
    }

    if meta_path.exists() and alice_x_path.exists() and bob_x_path.exists() and y_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta == target_meta:
                return alice_x_path, bob_x_path, y_path, meta
        except Exception:
            pass

    meta_path.write_text(json.dumps(target_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    alice_mm = np.memmap(alice_x_path, mode="w+", dtype=dtype, shape=(rows, alice_cols))
    bob_mm = np.memmap(bob_x_path, mode="w+", dtype=dtype, shape=(rows, bob_cols))
    y_mm = np.memmap(y_path, mode="w+", dtype=np.int8, shape=(rows,))

    rng = np.random.default_rng(seed)
    w_a = rng.normal(0, 1, size=(alice_cols,)).astype(np.float32)
    w_b = rng.normal(0, 1, size=(bob_cols,)).astype(np.float32)

    written = 0
    while written < rows:
        cur = min(chunk_rows, rows - written)
        a = rng.normal(0, 1, size=(cur, alice_cols)).astype(dtype, copy=False)
        b = rng.normal(0, 1, size=(cur, bob_cols)).astype(dtype, copy=False)
        logit = (a.astype(np.float32) @ w_a) + (b.astype(np.float32) @ w_b) + rng.normal(
            0, 0.5, size=(cur,)
        ).astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (p > 0.5).astype(np.int8)
        alice_mm[written : written + cur] = a
        bob_mm[written : written + cur] = b
        y_mm[written : written + cur] = y
        written += cur

    alice_mm.flush()
    bob_mm.flush()
    y_mm.flush()

    return alice_x_path, bob_x_path, y_path, target_meta


def build_heu_config(schema: str, bit_size: int, scale: int, mode: str) -> Dict:
    return {
        "sk_keeper": {"party": "alice"},
        "evaluators": [{"party": "bob"}],
        "mode": mode,
        "he_parameters": {
            "schema": schema,
            "key_pair": {"generate": {"bit_size": int(bit_size)}},
        },
        "encoding": {
            "cleartext_type": "DT_I32",
            "encoder": "BatchIntegerEncoder",
            "encoder_args": {"scale": int(scale)},
        },
    }


def make_fed_from_memmap(
    alice,
    bob,
    alice_x_path: Path,
    bob_x_path: Path,
    y_path: Path,
    rows: int,
    alice_cols: int,
    bob_cols: int,
    dtype_str: str,
):
    def load_x(path_str: str, r: int, c: int, dt: str):
        dt0 = _dtype_from_str(dt)
        return np.memmap(path_str, mode="r", dtype=dt0, shape=(r, c))

    def load_y(path_str: str, r: int):
        return np.memmap(path_str, mode="r", dtype=np.int8, shape=(r,))

    feature_data = FedNdarray(
        {
            alice: alice(load_x)(str(alice_x_path), rows, alice_cols, dtype_str),
            bob: bob(load_x)(str(bob_x_path), rows, bob_cols, dtype_str),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    label_data = FedNdarray(
        {alice: alice(load_y)(str(y_path), rows)},
        partition_way=PartitionWay.VERTICAL,
    )

    return feature_data, label_data


def subset_fed_indices(fed: FedNdarray, indices: np.ndarray) -> FedNdarray:
    parts = {}
    for pyu, part in fed.partitions.items():
        parts[pyu] = pyu(lambda a, idx: a[idx])(part.data, indices)
    return FedNdarray(parts, partition_way=fed.partition_way)


def subset_fed_head(fed: FedNdarray, n: int) -> FedNdarray:
    parts = {}
    for pyu, part in fed.partitions.items():
        parts[pyu] = pyu(lambda a, k: a[:k])(part.data, n)
    return FedNdarray(parts, partition_way=fed.partition_way)


def parse_sweep_rounds(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def run_plain_baseline_xgboost(
    alice_x_path: Path,
    bob_x_path: Path,
    y_path: Path,
    rows: int,
    alice_cols: int,
    bob_cols: int,
    dtype: np.dtype,
    baseline_rows: int,
    seed: int,
    num_boost_round: int,
    max_depth: int,
    learning_rate: float,
) -> Dict:
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError(f"xgboost not available: {e}") from e

    n = min(rows, int(baseline_rows))
    alice_x = np.memmap(alice_x_path, mode="r", dtype=dtype, shape=(rows, alice_cols))[:n]
    bob_x = np.memmap(bob_x_path, mode="r", dtype=dtype, shape=(rows, bob_cols))[:n]
    y = np.memmap(y_path, mode="r", dtype=np.int8, shape=(rows,))[:n].astype(np.int32, copy=False)
    x = np.concatenate([alice_x, bob_x], axis=1)

    dtrain = xgb.DMatrix(x, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "tree_method": "hist",
        "max_bin": 256,
        "seed": int(seed),
    }

    t0 = time.perf_counter()
    bst = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))
    t1 = time.perf_counter()

    pred = bst.predict(dtrain)
    auc = roc_auc_score(y, pred)
    return {"train_s": (t1 - t0), "auc": float(auc), "rows": int(n)}


def init_secretflow(object_store_gb: int):
    sf.shutdown()
    try:
        sf.init(
            ["alice", "bob"],
            address="local",
            _system_config={"lineage_pinning_enabled": False},
            object_store_memory=int(object_store_gb) * 1024 * 1024 * 1024,
        )
    except Exception:
        sf.shutdown()
        sf.init(
            address="local",
            _system_config={"lineage_pinning_enabled": False},
            object_store_memory=int(object_store_gb) * 1024 * 1024 * 1024,
        )


def build_cluster_def() -> Dict:
    return {
        "nodes": [
            {"party": "alice", "id": "local:0", "address": "127.0.0.1:12945"},
            {"party": "bob", "id": "local:1", "address": "127.0.0.1:12946"},
        ],
        "runtime_config": {
            "protocol": spu.spu_pb2.SEMI2K,
            "field": spu.spu_pb2.FM128,
        },
    }


def train_once(
    sgb: Sgb,
    params: Dict,
    feature_data: FedNdarray,
    label_data: FedNdarray,
    eval_x: FedNdarray,
    eval_y: FedNdarray,
    alice,
) -> Dict:
    t0 = time.perf_counter()
    model = sgb.train(dict(params), feature_data, label_data)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    yhat = model.predict(eval_x)
    yhat = reveal(yhat)
    t3 = time.perf_counter()

    y_eval = reveal(eval_y.partitions[alice])
    auc = roc_auc_score(y_eval, yhat)

    return {
        "train_s": t1 - t0,
        "predict_eval_s": t3 - t2,
        "auc_eval": float(auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000_000)
    parser.add_argument("--alice-cols", type=int, default=27)
    parser.add_argument("--bob-cols", type=int, default=27)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./bench_data")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=200_000)
    parser.add_argument("--object-store-gb", type=int, default=32)

    parser.add_argument("--heu-schema", type=str, default="ashe")
    parser.add_argument("--heu-bit-size", type=int, default=2048)
    parser.add_argument("--heu-scale", type=int, default=1)
    parser.add_argument("--heu-mode", type=str, default="PHEU")

    parser.add_argument("--num-boost-round", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-by-tree", type=float, default=0.9)
    parser.add_argument("--sketch-eps", type=float, default=0.08)
    parser.add_argument("--reg-lambda", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fixed-point-parameter", type=int, default=20)
    parser.add_argument("--sweep-rounds", type=str, default="")

    parser.add_argument("--baseline-xgboost", action="store_true")
    parser.add_argument("--baseline-rows", type=int, default=5_000_00)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dtype = _dtype_from_str(args.dtype)

    if args.generate_data:
        ensure_memmap_dataset(
            data_dir=data_dir,
            rows=args.rows,
            alice_cols=args.alice_cols,
            bob_cols=args.bob_cols,
            dtype=dtype,
            seed=args.seed,
            chunk_rows=args.chunk_rows,
        )

    alice_x_path = data_dir / "alice_x.dat"
    bob_x_path = data_dir / "bob_x.dat"
    y_path = data_dir / "y.dat"
    meta_path = data_dir / "meta.json"
    if not (alice_x_path.exists() and bob_x_path.exists() and y_path.exists() and meta_path.exists()):
        raise FileNotFoundError("dataset not found, run with --generate-data first")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = int(meta["rows"])
    alice_cols = int(meta["alice_cols"])
    bob_cols = int(meta["bob_cols"])

    init_secretflow(args.object_store_gb)

    cluster_def = build_cluster_def()
    alice = sf.PYU("alice")
    bob = sf.PYU("bob")

    heu_config = build_heu_config(
        schema=args.heu_schema,
        bit_size=args.heu_bit_size,
        scale=args.heu_scale,
        mode=args.heu_mode,
    )
    heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])
    sgb = Sgb(heu)

    feature_data, label_data = make_fed_from_memmap(
        alice=alice,
        bob=bob,
        alice_x_path=alice_x_path,
        bob_x_path=bob_x_path,
        y_path=y_path,
        rows=rows,
        alice_cols=alice_cols,
        bob_cols=bob_cols,
        dtype_str=args.dtype,
    )

    effective_rows = rows
    if args.train_rows and args.train_rows > 0 and args.train_rows < rows:
        effective_rows = int(args.train_rows)
        feature_data = subset_fed_head(feature_data, effective_rows)
        label_data = subset_fed_head(label_data, effective_rows)

    eval_rows = min(effective_rows, int(args.eval_rows))
    eval_x = subset_fed_head(feature_data, eval_rows)
    eval_y = subset_fed_head(label_data, eval_rows)

    base_params = {
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "sketch_eps": float(args.sketch_eps),
        "objective": "logistic",
        "reg_lambda": float(args.reg_lambda),
        "subsample": float(args.subsample),
        "colsample_by_tree": float(args.colsample_by_tree),
        "gamma": float(args.gamma),
        "seed": int(args.seed),
        "fixed_point_parameter": int(args.fixed_point_parameter),
        "base_score": 0.0,
    }

    sweep = parse_sweep_rounds(args.sweep_rounds)
    rounds_list = sweep if sweep else [int(args.num_boost_round)]

    results = []
    for r in rounds_list:
        if len(rounds_list) > 1:
            sf.shutdown()
            try:
                import ray

                ray.shutdown()
            except Exception:
                pass
            gc.collect()

            init_secretflow(args.object_store_gb)

            cluster_def = build_cluster_def()
            alice = sf.PYU("alice")
            bob = sf.PYU("bob")

            heu_config = build_heu_config(
                schema=args.heu_schema,
                bit_size=args.heu_bit_size,
                scale=args.heu_scale,
                mode=args.heu_mode,
            )
            heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])
            sgb = Sgb(heu)

            feature_data, label_data = make_fed_from_memmap(
                alice=alice,
                bob=bob,
                alice_x_path=alice_x_path,
                bob_x_path=bob_x_path,
                y_path=y_path,
                rows=rows,
                alice_cols=alice_cols,
                bob_cols=bob_cols,
                dtype_str=args.dtype,
            )

            if args.train_rows and args.train_rows > 0 and args.train_rows < rows:
                effective_rows = int(args.train_rows)
                feature_data = subset_fed_head(feature_data, effective_rows)
                label_data = subset_fed_head(label_data, effective_rows)
            else:
                effective_rows = rows

            eval_rows = min(effective_rows, int(args.eval_rows))
            eval_x = subset_fed_head(feature_data, eval_rows)
            eval_y = subset_fed_head(label_data, eval_rows)

        params = dict(base_params)
        params["num_boost_round"] = int(r)
        one = train_once(sgb, params, feature_data, label_data, eval_x, eval_y, alice)
        one["num_boost_round"] = int(r)
        results.append(one)
        gc.collect()

    sf.shutdown()
    try:
        import ray

        ray.shutdown()
    except Exception:
        pass
    gc.collect()

    out = {
        "secretflow_version": sf.__version__,
        "rows": effective_rows,
        "alice_cols": alice_cols,
        "bob_cols": bob_cols,
        "eval_rows": int(eval_rows),
        "heu_schema": args.heu_schema,
        "heu_bit_size": int(args.heu_bit_size),
        "heu_scale": int(args.heu_scale),
        "heu_mode": args.heu_mode,
        "sgb_params": base_params,
        "runs": results,
    }
    print(json.dumps(out, ensure_ascii=False))

    if args.baseline_xgboost:
        baseline_rounds = int(rounds_list[-1])
        base = run_plain_baseline_xgboost(
            alice_x_path=alice_x_path,
            bob_x_path=bob_x_path,
            y_path=y_path,
            rows=rows,
            alice_cols=alice_cols,
            bob_cols=bob_cols,
            dtype=dtype,
            baseline_rows=args.baseline_rows,
            seed=args.seed,
            num_boost_round=baseline_rounds,
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
        )
        ratio = results[-1]["train_s"] / max(base["train_s"], 1e-9)
        print(
            json.dumps(
                {
                    "baseline": base,
                    "train_time_ratio_secretflow_heu_over_plain": float(ratio),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
