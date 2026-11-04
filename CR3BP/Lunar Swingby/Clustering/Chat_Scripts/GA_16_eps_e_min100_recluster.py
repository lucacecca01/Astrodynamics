"""Try a minimum cluster size of 100 on the saved elliptic/SOI trial.

The 491-cluster result, integrated trajectories and all proposal partitions
are reused. STD eps/e/a/rp, radial compactness, direction and connectivity
retain their previous limits. Production code and previous results are kept.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from GA_16_eps_e_recluster import HERE, ROOT, SOURCE, load_helpers, load_integrated_data, summarize
from GA_16_eps_e_a_rp_recluster import complete_figures, size_summary
from GA_16_eps_e_a_rp_soi_recluster import PREVIOUS_WORK, TAG as PREVIOUS_TAG
from ga16_bridge_merge_acceleration import make_bridge_moment_prefiltered_merge
from ga16_eps_e_connectivity import ConnectivityShapeConstraint
from ga16_orbit_size_constraints import OrbitSizeConstraint, OrbitSizeGuidedConnectivity, augmented_validator_factory
from ga16_orbit_size_dispersion import individual_orbit_sizes, save_orbit_size_dispersion
from ga16_parking_orbit_filter import model_earth_soi, model_mu_earth, parking_orbit_mask
from ga16_refinement_checkpoint import refine_with_checkpoint
from ga16_size_recovery_acceleration import make_moment_prefiltered_recovery


FILTERED_WORK = ROOT / ".cache" / f"ga16_recluster_{PREVIOUS_TAG}"


def count_summary(labels, groups, minimum):
    counts = np.asarray([len(group) for group in groups])
    return dict(clusters=len(groups), assigned=int(np.count_nonzero(labels >= 0)),
                unassigned=int(np.count_nonzero(labels < 0)),
                clusters_below_minimum=int(np.count_nonzero(counts < minimum)),
                members_below_minimum=int(counts[counts < minimum].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.min_size < 1 or args.workers < 1:
        parser.error("Taglia minima e numero di processi devono essere positivi.")
    tag = f"{PREVIOUS_TAG}_min{args.min_size}"
    work = ROOT / ".cache" / f"ga16_recluster_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()
    catalog, full_ids, full_events = load_integrated_data()
    mu, soi = model_mu_earth(SOURCE), model_earth_soi(SOURCE)
    selected = np.flatnonzero(parking_orbit_mask(full_events[:, 0], mu, soi))
    ids, events = full_ids[selected], full_events[selected]
    with np.load(FILTERED_WORK / "result.npz") as old:
        if (not np.array_equal(old["source_ids"], ids)
                or not np.array_equal(old["selected_full_rows"], selected)):
            raise ValueError("Il risultato K491 non corrisponde al campione filtrato.")
        previous, previous_baseline = old["labels"], old["baseline"]
        previous_fingerprint = str(old["fingerprint"])
    features = np.column_stack((helpers["normalize_block"](events[:, 0:1]),
                                helpers["normalize_block"](events[:, 1:2])))
    sizes = np.column_stack(individual_orbit_sizes(events[:, 0], events[:, 1], mu))
    directions = events[:, 6]
    limits = np.array([.02, .04, np.inf])
    size_condition = OrbitSizeConstraint(sizes, 10000., 10000.)
    shape = OrbitSizeGuidedConnectivity(ConnectivityShapeConstraint(features, .05), size_condition)
    helpers["make_cluster_validator"] = augmented_validator_factory(helpers["make_cluster_validator"], size_condition)
    joint_limits = np.array([.02, .04, 10000., 10000.])
    observables = np.column_stack((events[:, :2], sizes))
    merge = make_bridge_moment_prefiltered_merge(
        helpers["merge_valid_clusters"], observables, directions, joint_limits, min_size=args.min_size,
    )
    recovery = make_moment_prefiltered_recovery(
        helpers["recover_unassigned_members"], observables, directions, joint_limits, min_size=args.min_size,
    )
    helpers["merge_valid_clusters"], helpers["recover_unassigned_members"] = merge, recovery
    valid = helpers["make_cluster_validator"](
        events[:, :3], features, limits, args.min_size, 2.5,
        directions=directions, shape_condition=shape,
    )
    previous_groups = helpers["cluster_groups"](previous)
    # Every original >=minimum cluster is mandatory seed coverage. Small
    # groups are proposals only and can be recovered through admissible unions.
    protected = np.full(len(ids), -1, dtype=np.int64)
    for cid, group in enumerate(g for g in previous_groups if len(g) >= args.min_size):
        if not valid(group):
            raise ValueError("Un cluster di partenza grande non rispetta i vincoli.")
        protected[group] = cid
    signature = hashlib.sha256(previous_fingerprint.encode())
    for value in (ids, events, features, sizes, np.array([args.min_size]), limits, joint_limits):
        signature.update(np.ascontiguousarray(value).tobytes())
    for path in (Path(__file__), SOURCE, HERE/"ga16_orbit_size_constraints.py",
                 HERE/"ga16_bridge_merge_acceleration.py", HERE/"ga16_size_merge_acceleration.py",
                 HERE/"ga16_size_recovery_acceleration.py", HERE/"ga16_refinement_checkpoint.py"):
        signature.update(path.read_bytes())
    fingerprint = signature.hexdigest()
    result_path = work / "result.npz"
    if result_path.exists():
        with np.load(result_path) as saved:
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError("La prova salvata usa codice o dati diversi.")
            labels, baseline = saved["labels"], saved["baseline"]
        seconds = json.loads((work/"clustering_summary.json").read_text())["clustering_seconds"]
        print("Risultato della prova riutilizzato.", flush=True)
    else:
        if args.plots_only:
            parser.error("La prova non e' stata ancora calcolata.")
        tick = time.monotonic()
        directory = FILTERED_WORK / "stages/fused"
        metadata = json.loads((directory/"metadata.json").read_text())
        if metadata["fingerprint"] != previous_fingerprint:
            raise ValueError("Il pool salvato non corrisponde al risultato K491.")
        members = np.load(directory/"members.npy", mmap_mode="r")
        offsets = np.load(directory/"offsets.npy")
        if offsets[-1] != len(members) or metadata["n_points"] != len(ids):
            raise ValueError("Pool di candidati non valido.")
        pool = list(np.split(members, offsets[1:-1])) + previous_groups
        with np.load(PREVIOUS_WORK/"result.npz") as old_eps:
            original_fingerprint = str(old_eps["fingerprint"])

        def reused_partition(features_arg, directions_arg, n_base, random_state=42):
            if features_arg is not features or directions_arg is not directions or n_base != 180:
                raise ValueError("Richiesta inattesa di una partizione.")
            with np.load(PREVIOUS_WORK/f"kmeans_K180_seed{random_state}.npz") as saved:
                if (str(saved["fingerprint"]) != original_fingerprint
                        or saved["labels"].shape != full_ids.shape):
                    raise ValueError("La partizione precedente usa dati diversi.")
                proposal = saved["labels"][selected]
            print(f"Seed {random_state}: partizione salvata rivalidata, nessun nuovo fit.", flush=True)
            return proposal

        helpers["kmeans_partition"] = reused_partition
        print(f"Prova minimo {args.min_size}: {len(ids):,} traiettorie, "
              f"{len(pool):,} candidati precedenti. Nessuna integrazione.", flush=True)
        labels, baseline, pool = refine_with_checkpoint(
            helpers, work/"stages", fingerprint,
            [previous_baseline, previous], events[:, :3], features, directions, limits,
            args.min_size, len(ids)//args.min_size, 2.5, seeds=(42, 7, 123), base_count=180,
            candidates=pool, previous_labels=protected, time_limit=30., shape_condition=shape,
        )
        if np.any(labels[protected >= 0] < 0):
            raise RuntimeError("La prova ha perso membri dei cluster grandi protetti.")
        if not all(valid(g) for g in helpers["cluster_groups"](labels)):
            raise RuntimeError("Un cluster non rispetta i vincoli.")
        seconds = time.monotonic()-tick
        pending = work/"result.tmp.npz"
        np.savez_compressed(pending, labels=labels, baseline=baseline, source_ids=ids,
                            selected_full_rows=selected, std_limits=limits,
                            size_std_limits_km=joint_limits[2:], earth_soi_km=soi,
                            minimum_size=args.min_size, fingerprint=fingerprint)
        pending.replace(result_path)
        (work/"clustering_summary.json").write_text(json.dumps(dict(clustering_seconds=seconds))+"\n")
        helpers["save_fusion_cache"](work/"candidates.npz", fingerprint, pool, labels)
        del pool
    groups = helpers["cluster_groups"](labels)
    if not all(valid(g) for g in groups) or np.any(labels[protected >= 0] < 0):
        raise RuntimeError("La verifica finale non e' superata.")
    output = HERE/"Clusters"/f"GA_16_plots_{tag}_K{len(groups)}"
    output.mkdir(parents=True, exist_ok=True)
    before = count_summary(previous, previous_groups, args.min_size)
    after = summarize(labels, events, helpers, limits)
    after.update(size_summary(labels, sizes, groups))
    after.update(count_summary(labels, groups, args.min_size))
    small = (previous >= 0) & (protected < 0)
    report = dict(before=before, after=after, minimum_size=args.min_size,
                  small_group_members_recovered=int(np.count_nonzero(small & (labels >= 0))),
                  small_group_members_unassigned=int(np.count_nonzero(small & (labels < 0))),
                  previously_unassigned_recovered=int(np.count_nonzero((previous < 0)&(labels >= 0))),
                  protected_large_group_members=int(np.count_nonzero(protected >= 0)),
                  protected_large_group_members_preserved=True,
                  earth_soi_km=soi, max_sigma=2.5, connectivity_radius=.05,
                  integration_calls=0, fresh_kmeans_runs=0, clustering_seconds=seconds,
                  fingerprint=fingerprint, previous_fingerprint=previous_fingerprint,
                  output=str(output), pair_prefilters=merge.prefilter_runs,
                  bridge_prefilters=merge.bridge_runs, recovery_prefilters=recovery.prefilter_runs)
    (output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.plots_only:
        helpers["save_cluster_velocity_statistics"](
            output/"cluster_velocity_statistics.txt", labels, events[:, 3:5], catalog[selected, 5:7],
            f"CR3BP_prova_GA_16_{tag}", catalog[selected, 7:9], directions,
        )
        exported = np.lib.format.open_memmap(output/"trajectory_exit_catalog.npy", mode="w+",
                                           dtype=np.float64, shape=(len(ids), 12))
        for first in range(0, len(ids), 100000):
            last = first+100000
            exported[first:last] = catalog[selected[first:last]]
            exported[first:last, 1] = labels[first:last]
        exported.flush()
        np.savetxt(output/"trajectory_exit_catalog.txt", exported, fmt=("%d", "%d")+("%.17e",)*10,
                   header="source_id cluster_id vx_soi_km_s vy_soi_km_s vz_soi_km_s t_perigee_days t_soi_days "
                          "x_soi_km y_soi_km eps_perigee_km2_s2 e_perigee omega_perigee_deg\n"
                          f"Source: {ROOT/'CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP_x.txt'}\n"
                          f"Selection: elliptic 0 < a <= {soi:.17e} km; STD eps < .02, e < .04, "
                          f"a/rp <= 10000 km; minimum cluster size {args.min_size}.")
        save_orbit_size_dispersion(exported, output)
    else:
        exported = np.load(output/"trajectory_exit_catalog.npy", mmap_mode="r")
    if not args.skip_plots:
        complete_figures(exported, output, [("Precedente minimo 10", previous),
                                          (f"Seed protetto minimo {args.min_size}", protected),
                                          (f"Risultato minimo {args.min_size}", labels)], args.workers)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
