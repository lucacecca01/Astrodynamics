"""Try tighter epsilon/eccentricity STD limits on the complete integrated catalog.

Reuses the K83 result and candidate pool as a warm start. No integration or
production-script main is executed; the original catalogs remain untouched.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]
SOURCE = HERE / "CR3BP_prova_GA_16_eps_e.py"
INPUT = HERE / "Clusters/GA_16_plots_eps_e_K83"
DATA = ROOT / ".cache/ga16_plot_recovery/GA_16_plots_eps_e_K83"
CANDIDATES = HERE / "Clusters/GA_16_eps_e_candidates.npz"


def load_helpers():
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    scope = dict(__file__=str(SOURCE))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


def load_integrated_data():
    catalog = np.load(DATA / "exit_catalog.npy", mmap_mode="r")
    ids = catalog[:, 0].astype(np.int64)
    events = np.empty((len(catalog), 7), dtype=float)
    events[:, :3] = catalog[:, 9:12]
    events[:, 3:6] = catalog[:, 2:5]
    current = DATA / "orbits"
    cache_ids = np.load(current / "source_ids.npy", mmap_mode="r")
    cache_features = np.load(current / "features.npy", mmap_mode="r")
    done = np.load(current / "done.npy", mmap_mode="r")
    rows = np.searchsorted(cache_ids, ids)
    matched = rows < len(cache_ids)
    matched[matched] &= cache_ids[rows[matched]] == ids[matched]
    assert done[rows[matched]].all()
    events[matched, 6] = cache_features[rows[matched], 6]
    missing = np.flatnonzero(~matched)
    if len(missing):
        previous = ROOT / ".cache/ga16_plot_recovery/GA_16_plots_vinf_time_K798/orbits"
        old_ids = np.load(previous / "source_ids.npy", mmap_mode="r")
        old_features = np.load(previous / "features.npy", mmap_mode="r")
        old_done = np.load(previous / "done.npy", mmap_mode="r")
        old_rows = np.searchsorted(old_ids, ids[missing])
        assert np.all(old_rows < len(old_ids)) and np.array_equal(old_ids[old_rows], ids[missing])
        assert old_done[old_rows].all()
        recovered = old_features[old_rows]
        assert np.allclose(recovered[:, :2], events[missing, :2], rtol=1e-9, atol=1e-8)
        assert np.allclose(recovered[:, 3:6], events[missing, 3:6], rtol=1e-9, atol=1e-8)
        assert np.allclose((recovered[:, 2]-events[missing, 2]+180)%360-180, 0, atol=1e-7)
        events[missing, 6] = recovered[:, 6]
    assert np.isfinite(events).all() and np.isin(events[:, 6], [-1, 1]).all()
    return catalog, ids, events


def summarize(labels, events, helpers, limits):
    from ga16_escape_dispersion import escape_dispersion
    groups = helpers["cluster_groups"](labels)
    sizes = np.array([len(g) for g in groups])
    stds = np.array([events[g, :2].std(axis=0) for g in groups])
    escape = np.array([escape_dispersion(events[g, 3:6]) for g in groups])
    return dict(std_limit_eps=float(limits[0]), std_limit_e=float(limits[1]),
                trajectories=len(labels), clusters=len(groups), assigned=int((labels >= 0).sum()),
                unassigned=int((labels < 0).sum()), min_size=int(sizes.min()),
                max_std_eps=float(stds[:, 0].max()), max_std_e=float(stds[:, 1].max()),
                weighted_mean_std_eps=float(np.average(stds[:, 0], weights=sizes)),
                weighted_mean_std_e=float(np.average(stds[:, 1], weights=sizes)),
                weighted_mean_std_speed_km_s=float(np.average(escape[:, 0], weights=sizes)),
                weighted_mean_direction_dispersion=float(np.average(escape[:, 1], weights=sizes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-eps", type=float, default=.02)
    parser.add_argument("--std-e", type=float, default=.04)
    parser.add_argument("--reference-connectivity", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true",
                        help="Completa le figure del risultato salvato, senza rifare clustering o cataloghi.")
    args = parser.parse_args()
    if not (np.isfinite(args.std_eps) and args.std_eps > 0 and np.isfinite(args.std_e) and args.std_e > 0):
        parser.error("Le STD massime devono essere finite e positive.")
    if args.workers < 1:
        parser.error("--workers deve essere positivo")
    tag = f"eps{args.std_eps:g}_e{args.std_e:g}".replace(".", "p")
    work = ROOT / ".cache" / f"ga16_recluster_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        from ga16_recluster_plot_completion import complete_plots
        with np.load(work / "result.npz") as saved:
            saved_labels = saved["labels"]
            count = len(np.unique(saved_labels[saved_labels >= 0]))
        output = HERE / "Clusters" / f"GA_16_plots_{tag}_K{count}"
        complete_plots(output, work, workers=args.workers)
        return
    helpers = load_helpers()
    catalog, ids, events = load_integrated_data()
    old_labels = catalog[:, 1].astype(np.int64)
    features = np.column_stack((helpers["normalize_block"](events[:, 0:1]),
                                helpers["normalize_block"](events[:, 1:2])))
    limits = np.array([args.std_eps, args.std_e, np.inf])
    if args.reference_connectivity:
        shape = helpers["ConnectivityShapeConstraint"](features, .05)
    else:
        from ga16_eps_e_connectivity import ConnectivityShapeConstraint
        shape = ConnectivityShapeConstraint(features, .05)
    valid = helpers["make_cluster_validator"](
        events[:, :3], features, limits, 10, 2.5, directions=events[:, 6], shape_condition=shape,
    )
    with np.load(CANDIDATES) as stored:
        assert np.array_equal(stored["labels"], old_labels)
        pool = list(np.split(stored["members"], stored["offsets"][1:-1]))
    # The first saved candidates are a complete, disjoint intermediate
    # partition. Revalidate it under the tighter bounds, as in production.
    base_labels = np.full(len(ids), -1, dtype=int)
    covered = 0
    for number, group in enumerate(pool):
        if np.any(base_labels[group] >= 0):
            break
        base_labels[group] = number
        covered += len(group)
        if covered == len(ids):
            break
    complete_base = covered == len(ids) and np.all(base_labels >= 0)
    partitions = [old_labels, base_labels] if complete_base else [old_labels]
    signature = hashlib.sha256()
    for array in (ids, events, features, limits):
        signature.update(np.ascontiguousarray(array).tobytes())
    signature.update(SOURCE.read_bytes())
    fingerprint = signature.hexdigest()
    started = time.monotonic()
    result = work / "result.npz"
    if result.exists():
        with np.load(result) as previous:
            if str(previous["fingerprint"]) != fingerprint:
                raise ValueError("La prova esistente ha dati o codice diversi; usare una nuova cartella.")
            labels = previous["labels"]
        print("Risultato della prova riutilizzato.", flush=True)
    else:
        print(f"Prova completa: {len(ids):,} traiettorie, {len(pool)} candidati precedenti, "
              f"STD eps < {limits[0]:g}, STD e < {limits[1]:g}; nessuna integrazione.", flush=True)
        original_kmeans = helpers["kmeans_partition"]

        def logged_kmeans(*values, **options):
            tick = time.monotonic()
            seed = values[3] if len(values) > 3 else options.get('random_state', 42)
            path = work / f"kmeans_K{values[2]}_seed{seed}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if str(saved["fingerprint"]) == fingerprint:
                        print(f"KMeans alternativo riutilizzato: K={values[2]}, seed={seed}.", flush=True)
                        return saved["labels"]
            print(f"KMeans alternativo: K={values[2]}, seed={seed}...", flush=True)
            computed = original_kmeans(*values, **options)
            np.savez_compressed(path, labels=computed, fingerprint=fingerprint)
            print(f"KMeans alternativo completato in {time.monotonic()-tick:.1f} s.", flush=True)
            return computed

        helpers["kmeans_partition"] = logged_kmeans
        labels, baseline, pool = helpers["refine_and_fuse_clusters"](
            partitions, events[:, :3], features, events[:, 6], limits, 10, max(1, len(ids)//100), 2.5,
            seeds=(42, 7, 123), base_count=180, candidates=pool, time_limit=30., shape_condition=shape,
        )
        groups = helpers["cluster_groups"](labels)
        assert all(valid(g) for g in groups)
        np.savez_compressed(result, labels=labels, baseline=baseline, source_ids=ids,
                            fingerprint=fingerprint, std_limits=limits)
        helpers["save_fusion_cache"](work / "candidates.npz", fingerprint, pool, labels)
    clustering_seconds = time.monotonic()-started
    groups = helpers["cluster_groups"](labels)
    assert all(valid(g) for g in groups)
    output = HERE / "Clusters" / f"GA_16_plots_{tag}_K{len(groups)}"
    output.mkdir(parents=True, exist_ok=True)
    before = summarize(old_labels, events, helpers, [.04, .05])
    after = summarize(labels, events, helpers, limits)
    report = dict(before=before, after=after, fingerprint=fingerprint, source_script=str(SOURCE),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  initialization="Warm start: saved complete intermediate partition, K83 and all saved candidates; production refinement/fusion/recovery with seeds 42,7,123.",
                  warm_start_base_clusters=int(np.max(base_labels)+1) if complete_base else None,
                  integration_calls=0, connectivity_radius=.05, max_sigma=2.5, minimum_size=10,
                  connectivity="reference BFS" if args.reference_connectivity else "exact grid radius graph",
                  additional_assigned=int(((old_labels < 0) & (labels >= 0)).sum()),
                  lost_assigned=int(((old_labels >= 0) & (labels < 0)).sum()),
                  clustering_seconds=clustering_seconds, output=str(output))
    (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(before))
        writer.writeheader(); writer.writerows((before, after))
    np.savez_compressed(output / "clustering_result.npz", labels=labels, source_ids=ids, std_limits=limits)
    helpers["save_cluster_velocity_statistics"](
        output / "cluster_velocity_statistics.txt", labels, events[:, 3:5], catalog[:, 5:7],
        f"CR3BP_prova_GA_16_eps_e_{tag}", catalog[:, 7:9], events[:, 6],
    )
    new_catalog = np.lib.format.open_memmap(output / "trajectory_exit_catalog.npy", mode="w+",
                                          dtype=np.float64, shape=catalog.shape)
    for start in range(0, len(catalog), 100000):
        stop = min(start+100000, len(catalog))
        new_catalog[start:stop] = catalog[start:stop]
        new_catalog[start:stop, 1] = labels[start:stop]
    new_catalog.flush()
    np.savetxt(output / "trajectory_exit_catalog.txt", new_catalog,
               fmt=("%d", "%d") + ("%.17e",)*10,
               header="source_id cluster_id vx_soi_km_s vy_soi_km_s vz_soi_km_s t_perigee_days t_soi_days "
                      "x_soi_km y_soi_km eps_perigee_km2_s2 e_perigee omega_perigee_deg\n"
                      f"Source: {ROOT / 'CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP_x.txt'}\n"
                      f"Clustering: STD eps < {limits[0]:g}, STD e < {limits[1]:g}; warm start K83; no time unimodality.")
    from ga16_escape_dispersion import recover_escape_rankings
    recover_escape_rankings(new_catalog, np.arange(len(groups)), groups, SOURCE, output)
    if not args.skip_plots:
        from ga16_recover_feature_plots import recover_feature_plots
        representatives = np.array([g[np.argmin(np.sum((features[g]-features[g].mean(axis=0))**2, axis=1))]
                                    for g in groups])
        recover_feature_plots(new_catalog, np.arange(len(groups)), groups, representatives, output,
                              workers=args.workers)
        from ga16_recluster_plot_completion import complete_plots
        complete_plots(output, work, workers=args.workers)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
