"""Try STD(a/rp)<=10,000 km, retaining the K196 eps/e constraints.

All integrated trajectories and KMeans starts are reused after validation.
The production script and earlier catalogs are never modified.
"""

import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from GA_16_eps_e_recluster import (
    CANDIDATES, DATA, HERE, ROOT, SOURCE, load_helpers, load_integrated_data, summarize,
)
from ga16_orbit_size_constraints import (
    OrbitSizeConstraint, OrbitSizeGuidedConnectivity, augmented_validator_factory,
)
from ga16_orbit_size_dispersion import (
    individual_orbit_sizes, model_mu_earth, save_orbit_size_dispersion,
)


PREVIOUS_WORK = ROOT / ".cache/ga16_recluster_eps0p02_e0p04"
PREVIOUS_OUTPUT = HERE / "Clusters/GA_16_plots_eps0p02_e0p04_K196"
_ORBIT_CONTEXT = None


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(SOURCE.read_bytes())
    return digest.hexdigest()


def candidate_pool(path):
    with np.load(path) as stored:
        return list(np.split(stored["members"], stored["offsets"][1:-1])), stored["labels"]


def base_partition(pool, n):
    labels = np.full(n, -1, dtype=np.int64)
    covered = 0
    for number, group in enumerate(pool):
        if np.any(labels[group] >= 0):
            break
        labels[group] = number
        covered += len(group)
        if covered == n:
            return labels
    raise ValueError("La partizione iniziale completa non e' ricostruibile.")


def size_summary(labels, sizes, groups):
    stds = np.array([sizes[g].std(axis=0) for g in groups])
    counts = [len(g) for g in groups]
    return dict(max_std_a_km=float(stds[:, 0].max()), max_std_rp_km=float(stds[:, 1].max()),
                median_std_a_km=float(np.median(stds[:, 0])), median_std_rp_km=float(np.median(stds[:, 1])),
                weighted_mean_std_a_km=float(np.average(stds[:, 0], weights=counts)),
                weighted_mean_std_rp_km=float(np.average(stds[:, 1], weights=counts)))


def orbit_pages(task):
    from ga16_recover_orbit_plots import render_orbit_plots
    cache, output, labels, ids, counts, representatives = _ORBIT_CONTEXT
    frame, pages = task
    paths = render_orbit_plots(cache, output, labels, ids, counts, representatives,
                               render_representatives=False, frame_filter={frame}, page_filter=set(pages))
    print("Pagine completate: "+", ".join(p.name for p in paths), flush=True)
    return [str(p) for p in paths]


def complete_figures(catalog, output, stages, workers):
    from GA_16_recover_plots import expected_plot_names, prepare_groups
    from ga16_catalog_orbit_view import open_catalog_orbit_view
    from ga16_escape_dispersion import recover_escape_rankings
    from ga16_recluster_diagnostics import recover_diagnostics
    from ga16_recover_feature_plots import recover_feature_plots, valid_png
    from ga16_recover_orbit_plots import render_orbit_plots

    ids, labels, cluster_ids, counts, groups, representatives = prepare_groups(
        catalog, output / "cluster_velocity_statistics.txt", SOURCE,
    )
    recover_feature_plots(catalog, cluster_ids, groups, representatives, output, workers=min(workers, 4))
    if not (output/f"orbital_dispersion_ranking_K{len(groups)}.pdf").exists():
        recover_escape_rankings(catalog, cluster_ids, groups, SOURCE, output)
    recover_diagnostics(catalog, stages, output)
    cache = open_catalog_orbit_view(catalog, SOURCE)
    scope = load_helpers()
    scope.update(SAVE=True, PLOT_CLUSTERS=False, output_directory=output, mu=cache.context["mu"])
    if not valid_png(output / "unassigned_trajectories.png"):
        scope["plot_unassigned_trajectories"](cache.curves, labels, cache.context["mu"])
    render_orbit_plots(cache, output, labels, cluster_ids, counts, ids[representatives], render_clusters=False)
    global _ORBIT_CONTEXT
    _ORBIT_CONTEXT = (cache, output, labels, cluster_ids, counts, ids[representatives])
    pages = (len(groups)+23)//24
    lanes = min(workers, pages)
    tasks = [(frame, list(range(lane, pages, lanes)))
             for frame in (False, True) for lane in range(lanes)]
    if workers == 1:
        for task in tasks:
            orbit_pages(task)
    else:
        with get_context("fork").Pool(workers) as pool:
            list(pool.imap_unordered(orbit_pages, tasks))
    _ORBIT_CONTEXT = None
    expected = expected_plot_names(cluster_ids)+[
        "unassigned_trajectories.png", "STD_vs_K.png", "gaussian_shape_vs_stage.png",
        "physical_STD_and_3sigma_vs_K.png", f"cluster_diagnostics_K{len(groups)}.png",
        f"orbital_size_dispersion_K{len(groups)}.png",
    ]
    missing = [name for name in expected if not valid_png(output/name)]
    report = dict(expected_pngs=len(expected), missing_or_invalid=missing,
                  integration_calls=0, assigned_orbits=int(np.count_nonzero(labels >= 0)))
    (output / "plot_completion.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if missing:
        raise RuntimeError("Alcuni plot non sono stati completati.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-a-km", type=float, default=10000.)
    parser.add_argument("--std-rp-km", type=float, default=10000.)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if (args.workers < 1 or not np.isfinite([args.std_a_km, args.std_rp_km]).all()
            or min(args.std_a_km, args.std_rp_km) <= 0):
        parser.error("Limiti STD e numero di processi devono essere positivi.")
    tag = f"eps0p02_e0p04_a{args.std_a_km:g}_rp{args.std_rp_km:g}".replace(".", "p")
    work = ROOT / ".cache" / f"ga16_recluster_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()
    catalog, ids, events = load_integrated_data()
    features = np.column_stack((helpers["normalize_block"](events[:, 0:1]),
                                helpers["normalize_block"](events[:, 1:2])))
    limits = np.array([.02, .04, np.inf])
    previous_fingerprint = array_fingerprint(ids, events, features, limits)
    with np.load(PREVIOUS_WORK / "result.npz") as stored:
        if str(stored["fingerprint"]) != previous_fingerprint:
            raise ValueError("I dati e il codice non corrispondono alla prova K196 salvata.")
        previous = stored["labels"]
        previous_baseline = stored["baseline"]
    sizes = np.column_stack(individual_orbit_sizes(events[:, 0], events[:, 1], model_mu_earth()))
    directions = events[:, 6]
    constraint = OrbitSizeConstraint(sizes, args.std_a_km, args.std_rp_km)
    from ga16_eps_e_connectivity import ConnectivityShapeConstraint
    shape = OrbitSizeGuidedConnectivity(ConnectivityShapeConstraint(features, .05), constraint)
    original_factory = helpers["make_cluster_validator"]
    helpers["make_cluster_validator"] = augmented_validator_factory(original_factory, constraint)
    from ga16_bridge_merge_acceleration import make_bridge_moment_prefiltered_merge
    from ga16_size_recovery_acceleration import make_moment_prefiltered_recovery
    observables = np.column_stack((events[:, :2], sizes))
    joint_limits = np.r_[limits[:2], constraint.limits]
    merge = make_bridge_moment_prefiltered_merge(
        helpers["merge_valid_clusters"], observables,
        directions, joint_limits, min_size=10,
    )
    helpers["merge_valid_clusters"] = merge
    recovery = make_moment_prefiltered_recovery(
        helpers["recover_unassigned_members"], observables, directions, joint_limits, min_size=10,
    )
    helpers["recover_unassigned_members"] = recovery
    valid = helpers["make_cluster_validator"](
        events[:, :3], features, limits, 10, 2.5, directions=events[:, 6], shape_condition=shape,
    )
    signature = hashlib.sha256(previous_fingerprint.encode())
    signature.update(np.asarray(constraint.limits).tobytes())
    signature.update(previous.tobytes())
    for source in (Path(__file__), HERE / "ga16_orbit_size_constraints.py",
                   HERE / "ga16_size_merge_acceleration.py", HERE / "ga16_bridge_merge_acceleration.py",
                   HERE / "ga16_size_recovery_acceleration.py", HERE / "ga16_refinement_checkpoint.py"):
        signature.update(source.read_bytes())
    fingerprint = signature.hexdigest()
    result_path = work / "result.npz"
    tick = time.monotonic()
    if result_path.exists():
        with np.load(result_path) as stored:
            if str(stored["fingerprint"]) != fingerprint:
                raise ValueError("Codice o dati della prova cambiati: usare una nuova cartella.")
            labels, baseline = stored["labels"], stored["baseline"]
        seconds = json.loads((work / "clustering_summary.json").read_text())["clustering_seconds"]
        print("Risultato a/rp salvato riutilizzato.", flush=True)
    else:
        if args.plots_only:
            parser.error("La prova non e' ancora stata calcolata.")
        old_pool, old_labels = candidate_pool(CANDIDATES)
        if not np.array_equal(old_labels, catalog[:, 1]):
            raise ValueError("Le etichette del pool K83 sono cambiate.")
        base = base_partition(old_pool, len(catalog))
        pool, pool_labels = candidate_pool(PREVIOUS_WORK / "candidates.npz")
        if not np.array_equal(pool_labels, previous):
            raise ValueError("Le etichette del pool K196 sono cambiate.")
        pool.extend(old_pool)
        del old_pool

        def reused_kmeans(features_arg, directions_arg, n_base, random_state=42):
            if features_arg is not features or directions_arg is not directions or n_base != 180:
                raise ValueError("La partizione KMeans salvata usa feature o parametri diversi.")
            path = PREVIOUS_WORK / f"kmeans_K{n_base}_seed{random_state}.npz"
            with np.load(path) as stored:
                if str(stored["fingerprint"]) != previous_fingerprint:
                    raise ValueError("Le feature del KMeans salvato sono diverse.")
                values = stored["labels"]
            if values.shape != previous.shape:
                raise ValueError("Dimensione della partizione KMeans incompatibile.")
            print(f"KMeans K={n_base}, seed={random_state}: riutilizzato, feature identiche.", flush=True)
            return values

        helpers["kmeans_partition"] = reused_kmeans
        print(f"Prova completa su {len(ids):,} righe: STD a <= {args.std_a_km:g} km, "
              f"STD rp <= {args.std_rp_km:g} km; STD eps < .02, STD e < .04.\n"
              "Nessuna integrazione; nessun membro eliminato preventivamente.", flush=True)
        from ga16_refinement_checkpoint import refine_with_checkpoint
        labels, baseline, pool = refine_with_checkpoint(
            helpers, work / "stages", fingerprint,
            [old_labels, base, previous_baseline, previous], events[:, :3], features,
            directions, limits, 10, len(ids)//100, 2.5,
            seeds=(42, 7, 123), base_count=180, candidates=pool, time_limit=30., shape_condition=shape,
        )
        groups = helpers["cluster_groups"](labels)
        if not all(valid(group) for group in groups):
            raise RuntimeError("Un cluster non rispetta i vincoli finali.")
        seconds = time.monotonic()-tick
        temporary = work / "result.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, labels=labels, baseline=baseline, source_ids=ids,
                                std_limits=limits, size_std_limits_km=constraint.limits,
                                fingerprint=fingerprint)
        temporary.replace(result_path)
        (work / "clustering_summary.json").write_text(json.dumps(dict(clustering_seconds=seconds))+"\n")
        helpers["save_fusion_cache"](work / "candidates.npz", fingerprint, pool, labels)
        del pool
    groups = helpers["cluster_groups"](labels)
    if not all(valid(group) for group in groups):
        raise RuntimeError("Un cluster salvato non rispetta i vincoli.")
    output = HERE / "Clusters" / f"GA_16_plots_{tag}_K{len(groups)}"
    output.mkdir(parents=True, exist_ok=True)
    previous_groups = helpers["cluster_groups"](previous)
    before = summarize(previous, events, helpers, limits)
    before.update(size_summary(previous, sizes, previous_groups))
    after = summarize(labels, events, helpers, limits)
    after.update(size_summary(labels, sizes, groups))
    report = dict(before=before, after=after, size_std_limits_km=constraint.limits.tolist(),
                  added_assignments=int(np.count_nonzero((previous < 0) & (labels >= 0))),
                  lost_assignments=int(np.count_nonzero((previous >= 0) & (labels < 0))),
                  minimum_size=10, max_sigma=2.5, eps_e_connectivity_radius=.05,
                  clustering_seconds=seconds, integration_calls=0, fresh_kmeans_runs=0,
                  fingerprint=fingerprint, output=str(output),
                  pair_prefilters=merge.prefilter_runs,
                  bridge_prefilters=merge.bridge_runs, recovery_prefilters=recovery.prefilter_runs,
                  split_guidance="Exact 1-D two-means on the most violated a/rp STD; original eps/e fallback.")
    (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.plots_only:
        helpers["save_cluster_velocity_statistics"](
            output / "cluster_velocity_statistics.txt", labels, events[:, 3:5], catalog[:, 5:7],
            f"CR3BP_prova_GA_16_{tag}", catalog[:, 7:9], events[:, 6],
        )
        new_catalog = np.lib.format.open_memmap(output / "trajectory_exit_catalog.npy", mode="w+",
                                              dtype=np.float64, shape=catalog.shape)
        for first in range(0, len(catalog), 100000):
            stop = first+100000
            new_catalog[first:stop] = catalog[first:stop]
            new_catalog[first:stop, 1] = labels[first:stop]
        new_catalog.flush()
        np.savetxt(output / "trajectory_exit_catalog.txt", new_catalog,
                   fmt=("%d", "%d")+("%.17e",)*10,
                   header="source_id cluster_id vx_soi_km_s vy_soi_km_s vz_soi_km_s t_perigee_days t_soi_days "
                          "x_soi_km y_soi_km eps_perigee_km2_s2 e_perigee omega_perigee_deg\n"
                          f"Source: {ROOT / 'CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP_x.txt'}\n"
                          f"STD eps < .02; STD e < .04; STD a <= {args.std_a_km:g} km; "
                          f"STD rp <= {args.std_rp_km:g} km; individual osculating elements, ddof=0.")
        size_report = save_orbit_size_dispersion(new_catalog, output)
        (work / "size_summary.json").write_text(json.dumps(size_report, indent=2)+"\n")
    else:
        new_catalog = np.load(output / "trajectory_exit_catalog.npy", mmap_mode="r")
    if not args.skip_plots:
        complete_figures(new_catalog, output,
                         [("K196 precedente", previous), ("Split con limiti a/rp", baseline),
                          ("Fusione finale", labels)], args.workers)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
