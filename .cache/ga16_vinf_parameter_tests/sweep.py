"""Run the production velocity clustering on frozen, full-database features."""
import argparse
import ast
import contextlib
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "CR3BP/Lunar Swingby/Clustering/CR3BP_prova_GA_16_vinf.py"
CACHE = SOURCE.parent / "Clusters/GA_16_vxvy_candidates.npz"
DATA = ROOT / ".cache/ga16_sigma_recovery_tests/features.npz"
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_app():
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    app = {"__file__": str(SOURCE)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), app)
    return app


def groups_from_cache(cache):
    return list(np.split(cache["members"], cache["offsets"][1:-1]))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, nargs="+", default=[2, 2.5, 3])
    parser.add_argument("--std", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--min-size", type=int, nargs="+", default=[10])
    parser.add_argument("--bases", type=int, nargs=2, default=[99, 180])
    args = parser.parse_args()
    frozen_initial = OUT / "initial_candidates.npz"
    if args.bases != [99, 180]:
        OUT = OUT / f"bases{args.bases[0]}_{args.bases[1]}"
        OUT.mkdir(exist_ok=True)
    app = load_app()
    with np.load(DATA) as data:
        events = data["features"].copy()
        source_ids = data["source_ids"].copy()
        n_selected = len(data["selected_ids"])
        n_preexcluded = len(data["rejected_ids"])
    physical = events[:, 3:6]
    features = app["normalize_block"](physical)
    directions = events[:, 6]
    n = len(events)
    cap = n // 100
    assert np.all(physical[:, 2] == 0), "This experiment requires the planar dataset."
    database = np.loadtxt(ROOT / "CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP.txt", skiprows=1)
    fingerprint = app["fusion_fingerprint"](source_ids, database[source_ids], events, features)
    initial = OUT / "initial_candidates.npz"
    if not initial.exists():
        with np.load(frozen_initial if frozen_initial.exists() else CACHE) as data:
            np.savez_compressed(initial, **{key: data[key] for key in data.files})
    with np.load(initial) as data:
        assert str(data["fingerprint"]) == fingerprint
        candidates = groups_from_cache(data)
    metadata = dict(source_sha256=digest(SOURCE), data_fingerprint=fingerprint,
                    initial_cache_sha256=digest(initial), n_clustering=n,
                    n_selected=n_selected, n_database=len(database), preexcluded=n_preexcluded,
                    max_clusters=cap, base_min=200, bases=args.bases, seeds=[42, 7, 123],
                    dip_alpha=0.01, gap_fraction=0.2, strong_gap_fraction=0.5,
                    outlier_sigma=3, gaussian_qq="disabled", fusion_time_limit=30,
                    cache_policy="same frozen initial pool for every configuration",
                    objective="maximize coverage, then minimize cluster count")
    metadata_path = OUT / "metadata.json"
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text()) == metadata, "Inputs or implementation changed."
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    original_milp = app["milp"]
    solver_results = []

    def recorded_milp(*a, **kw):
        result = original_milp(*a, **kw)
        solver_results.append(dict(status=int(result.status), message=str(result.message),
                                   gap=None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)))
        return result

    app["milp"] = recorded_milp
    original_kmeans = app["kmeans_partition"]

    def cached_kmeans(x, d, count, random_state=42):
        path = OUT / f"kmeans_{count}_{random_state}.npz"
        if path.exists():
            with np.load(path) as saved:
                return saved["labels"].copy()
        labels = original_kmeans(x, d, count, random_state)
        np.savez_compressed(path, labels=labels)
        return labels

    app["kmeans_partition"] = cached_kmeans
    print(f"Full data: {n}; excluded before clustering: {n_preexcluded}; fixed initial candidates: {len(candidates)}", flush=True)
    results_path = OUT / "comparison.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    for sigma in args.sigma:
        sigma_tag = str(sigma).replace(".", "p")
        base_path = OUT / f"base_sigma_{sigma_tag}.npz"
        base_start = time.monotonic()
        if base_path.exists():
            with np.load(base_path) as data:
                partitions = [data["base0"], data["base1"]]
        else:
            partitions = []
            labels = None
            with (OUT / f"base_sigma_{sigma_tag}.log").open("w") as log, contextlib.redirect_stdout(log):
                for count in args.bases:
                    try:
                        labels = app["constrained_clustering"](
                            features, directions, count, min_size=200,
                            sigma_limit=sigma, previous_labels=labels)
                        partitions.append(labels.copy())
                    except app["NoFeasiblePartitionError"] as exc:
                        print(f"Fallback: {exc}", flush=True)
                        partitions.append(cached_kmeans(features, directions, count))
                        labels = None
            np.savez_compressed(base_path, base0=partitions[0], base1=partitions[1])
        print(f"Bases sigma={sigma} ready in {time.monotonic()-base_start:.1f}s", flush=True)
        for std in args.std:
            for min_size in args.min_size:
                name = f"std{std:g}_sigma{sigma:g}_min{min_size}"
                if any(row["name"] == name for row in rows):
                    continue
                print(f"START {name}", flush=True)
                start = time.monotonic()
                limits = np.array([std, std, np.inf])
                shape = app["ClusterShapeConstraint"](
                    physical, limits, dip_alpha=0.01, gap_fraction=0.2,
                    strong_gap_fraction=0.5, outlier_sigma=3)
                valid = app["make_cluster_validator"](
                    physical, features, limits, min_size, sigma,
                    directions=directions, shape_condition=shape)
                solver_results.clear()
                with (OUT / f"{name}.log").open("w") as log, contextlib.redirect_stdout(log):
                    labels, baseline, pool = app["refine_and_fuse_clusters"](
                        partitions, physical, features, directions, limits, min_size, cap, sigma,
                        seeds=(42, 7, 123), base_count=args.bases[-1], candidates=candidates,
                        time_limit=30, shape_condition=shape)
                groups = app["cluster_groups"](labels)
                assert len(groups) <= cap and all(valid(ids) for ids in groups)
                assert sum(map(len, groups)) == np.count_nonzero(labels >= 0)
                stds = np.array([physical[ids].std(axis=0)[:2] for ids in groups])
                excluded = int(np.count_nonzero(labels < 0))
                row = dict(name=name, std_vx=std, std_vy=std, max_sigma=sigma,
                           min_size=min_size, clusters=len(groups), excluded=excluded,
                           excluded_pct=100*excluded/n,
                           total_excluded_selected=excluded+n_preexcluded,
                           total_excluded_selected_pct=100*(excluded+n_preexcluded)/n_selected,
                           total_excluded_database_pct=100*(excluded+len(database)-n)/len(database),
                           min_actual_size=min(map(len, groups), default=0),
                           mean_std_vx=float(stds[:, 0].mean()), mean_std_vy=float(stds[:, 1].mean()),
                           max_std_vx=float(stds[:, 0].max()), max_std_vy=float(stds[:, 1].max()),
                           candidate_count=len(pool), runtime_s=time.monotonic()-start,
                           solver_status=solver_results[-1]["status"], solver_gap=solver_results[-1]["gap"])
                app["save_fusion_cache"](OUT / f"{name}.npz", fingerprint, pool, labels)
                np.savez_compressed(OUT / f"{name}_baseline.npz", labels=baseline)
                rows.append(row)
                results_path.write_text(json.dumps(rows, indent=2) + "\n")
                with (OUT / "comparison.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(row), flush=True)
    print(f"DONE. Production source unchanged: {digest(SOURCE) == metadata['source_sha256']}", flush=True)


if __name__ == "__main__":
    main()
