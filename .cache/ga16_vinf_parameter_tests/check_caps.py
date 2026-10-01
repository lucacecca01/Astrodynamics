"""Tune the existing cluster cap on the best tested pools; no algorithm changes."""
import contextlib
import csv
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from sweep import DATA, OUT, SOURCE, digest, groups_from_cache, load_app


def main():
    app = load_app()
    metadata = json.loads((OUT / "metadata.json").read_text())
    source = SOURCE.read_bytes().decode()
    if digest(SOURCE) != metadata["source_sha256"]:
        # Allow parameter edits, but still require byte-identical implementation.
        lines = source.splitlines(keepends=True)
        originals = {"MAX_SIGMA": "2", "MIN_CLUSTER_SIZE": "10",
                     "STD_LIMITS": "np.array([0.05, 0.05, np.inf])",
                     "BASE_CLUSTER_COUNTS": "(99, 180)", "BASE_MIN_CLUSTER_SIZE": "200"}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in originals:
                    value = node.value
                    assert value.lineno == value.end_lineno
                    line = lines[value.lineno-1]
                    lines[value.lineno-1] = line[:value.col_offset] + originals[name] + line[value.end_col_offset:]
        assert hashlib.sha256("".join(lines).encode()).hexdigest() == metadata["source_sha256"], \
            "Implementation changed beyond the clustering parameters."
        print("Verified: only global parameters changed; tested functions are identical.", flush=True)
    records = []
    for path in sorted(OUT.glob("**/comparison.json")):
        if not (path.parent / "metadata.json").exists():
            continue
        settings = json.loads((path.parent / "metadata.json").read_text())
        assert settings["data_fingerprint"] == metadata["data_fingerprint"]
        for row in json.loads(path.read_text()):
            records.append(dict(row, bases="/".join(map(str, settings["bases"])),
                                directory=str(path.parent.relative_to(OUT))))
    with (OUT / "all_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with np.load(DATA) as data:
        physical = data["features"][:, 3:6]
        directions = data["features"][:, 6]
        source_ids = data["source_ids"]
    features = app["normalize_block"](physical)
    n = len(features)
    original_milp = app["milp"]
    results = []

    def milp(*args, **kwargs):
        result = original_milp(*args, **kwargs)
        results.append(result)
        return result

    app["milp"] = milp
    output = OUT / "cap_tests"
    output.mkdir(exist_ok=True)
    summaries = []
    for std in (0.1, 0.2, 0.3):
        eligible = [r for r in records if r["std_vx"] == std and r["total_excluded_selected_pct"] < 1]
        row = min(eligible, key=lambda r: (r["clusters"], r["excluded"]))
        directory = OUT / row["directory"]
        with np.load(directory / (row["name"] + ".npz")) as saved:
            pool = groups_from_cache(saved)
            original_labels = saved["labels"]
        limits = np.array([std, std, np.inf])
        shape = app["ClusterShapeConstraint"](physical, limits, dip_alpha=.01, gap_fraction=.2,
                                              strong_gap_fraction=.5, outlier_sigma=3)
        validator = app["make_cluster_validator"](
            physical, features, limits, row["min_size"], row["max_sigma"],
            directions=directions, shape_condition=shape)
        pool = [ids for ids in pool if validator(ids)]
        valid_keys = {np.asarray(ids, dtype=np.int64).tobytes() for ids in pool}

        def valid(ids):
            return np.asarray(ids, dtype=np.int64).tobytes() in valid_keys

        trials = {row["clusters"]: (row["excluded"], original_labels)}
        log_name = f"std{std:g}"
        print(f"CAP TEST {log_name}: bases={row['bases']}, sigma={row['max_sigma']}, "
              f"min_size={row['min_size']}, initial_K={row['clusters']}", flush=True)

        def evaluate(cap):
            if cap not in trials:
                with (output / f"{log_name}_cap{cap}.log").open("w") as log, contextlib.redirect_stdout(log):
                    labels, _ = app["fuse_cluster_candidates"](pool, n, valid, cap, time_limit=60)
                assert results[-1].status == 0, "Cannot certify infeasibility after a solver timeout."
                groups = app["cluster_groups"](labels)
                assert len(groups) <= cap and all(validator(ids) for ids in groups)
                trials[cap] = (int(np.count_nonzero(labels < 0)), labels)
                print(f"  cap={cap}: K={len(groups)}, excluded={trials[cap][0]}", flush=True)
            return trials[cap]

        for basis, excluded_budget in (
            ("clustering", int(np.ceil(.01*n)) - 1),
            ("including_preexcluded", int(np.ceil(.01*metadata["n_selected"])) - 1 - metadata["preexcluded"]),
        ):
            low, high = 0, row["clusters"]
            while high - low > 1:
                middle = (low + high) // 2
                excluded, labels = evaluate(middle)
                if excluded <= excluded_budget:
                    high = middle
                else:
                    low = middle
            excluded, labels = evaluate(high)
            groups = app["cluster_groups"](labels)
            if high > 1:
                assert evaluate(high-1)[0] > excluded_budget
            actual_stds = np.array([physical[ids].std(axis=0)[:2] for ids in groups])
            summary = dict(std=std, max_sigma=row["max_sigma"], min_size=row["min_size"],
                           bases=row["bases"], target_basis=basis, max_clusters=high,
                           clusters=len(groups), excluded=excluded, excluded_pct=100*excluded/n,
                           total_excluded_selected_pct=100*(excluded+metadata["preexcluded"])/metadata["n_selected"],
                           max_std_vx=float(actual_stds[:, 0].max()), max_std_vy=float(actual_stds[:, 1].max()),
                           mean_std_vx=float(actual_stds[:, 0].mean()), mean_std_vy=float(actual_stds[:, 1].mean()),
                           source_result=str(directory.relative_to(OUT) / (row["name"] + ".npz")),
                           min_actual_size=min(map(len, groups)))
            summaries.append(summary)
            np.savez_compressed(output / f"{log_name}_{basis}.npz", labels=labels, source_ids=source_ids)
            print(json.dumps(summary), flush=True)
            (output / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
            with (output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary))
                writer.writeheader()
                writer.writerows(summaries)
    print(f"FINISHED: {len(records)} full-pipeline configurations; {len(summaries)} cap optima.", flush=True)


if __name__ == "__main__":
    main()
