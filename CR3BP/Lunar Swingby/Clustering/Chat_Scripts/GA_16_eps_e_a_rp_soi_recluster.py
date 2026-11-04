"""Recluster integrated elliptic perigee orbits with 0 < a <= Earth SOI.

Retains STD eps < .02, STD e < .04, STD a/rp <= 10,000 km. Existing
partitions are only proposals, trimmed and revalidated with the filtered
population's normalization. No new integration or KMeans fitting is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from GA_16_eps_e_recluster import CANDIDATES, HERE, ROOT, SOURCE, load_helpers, load_integrated_data, summarize
from GA_16_eps_e_a_rp_recluster import candidate_pool, base_partition, complete_figures, size_summary
from ga16_bridge_merge_acceleration import make_bridge_moment_prefiltered_merge
from ga16_eps_e_connectivity import ConnectivityShapeConstraint
from ga16_orbit_size_constraints import OrbitSizeConstraint, OrbitSizeGuidedConnectivity, augmented_validator_factory
from ga16_orbit_size_dispersion import individual_orbit_sizes, save_orbit_size_dispersion
from ga16_parking_orbit_filter import model_earth_soi, model_mu_earth, parking_orbit_mask
from ga16_refinement_checkpoint import refine_with_checkpoint
from ga16_size_recovery_acceleration import make_moment_prefiltered_recovery


PREVIOUS_WORK = ROOT / ".cache/ga16_recluster_eps0p02_e0p04"
SIZE_WORK = ROOT / ".cache/ga16_recluster_eps0p02_e0p04_a10000_rp10000"
SNAPSHOT = ROOT / ".cache/ga16_source_snapshots/CR3BP_prova_GA_16_eps_e_b912cea81964d4b7b8de7fb4d41009fdfb70b7ef60f8c139cf99ce65816ef5dd.py"
TAG = "eps0p02_e0p04_a10000_rp10000_elliptic_within_soi"


def trimmed_pool(pool, lookup):
    result = []
    for members in pool:
        rows = lookup[members]
        rows = rows[rows >= 0]
        if len(rows):
            result.append(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers deve essere positivo")
    work = ROOT / ".cache" / f"ga16_recluster_{TAG}"
    work.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()
    full_catalog, full_ids, full_events = load_integrated_data()
    full_features = np.column_stack((helpers["normalize_block"](full_events[:, 0:1]),
                                     helpers["normalize_block"](full_events[:, 1:2])))
    limits = np.array([.02, .04, np.inf])
    previous_signature = hashlib.sha256()
    for array in (full_ids, full_events, full_features, limits):
        previous_signature.update(np.ascontiguousarray(array).tobytes())
    previous_signature.update(SNAPSHOT.read_bytes())
    previous_fingerprint = previous_signature.hexdigest()
    with np.load(PREVIOUS_WORK / "result.npz") as stored:
        if (str(stored["fingerprint"]) != previous_fingerprint
                or not np.array_equal(stored["source_ids"], full_ids)):
            raise ValueError("Le partizioni precedenti non corrispondono ai dati integrati.")
        previous_eps = stored["labels"]
    with np.load(SIZE_WORK / "result.npz") as stored:
        if not np.array_equal(stored["source_ids"], full_ids):
            raise ValueError("Gli identificativi della prova a/rp sono diversi.")
        previous_sizes = stored["labels"]
    mu, soi = model_mu_earth(SOURCE), model_earth_soi(SOURCE)
    keep = parking_orbit_mask(full_events[:, 0], mu, soi)
    selected = np.flatnonzero(keep)
    if not len(selected):
        raise ValueError("Nessuna orbita ellittica con a entro la SOI terrestre.")
    ids, events = full_ids[selected], full_events[selected]
    features = np.column_stack((helpers["normalize_block"](events[:, 0:1]),
                                helpers["normalize_block"](events[:, 1:2])))
    sizes = np.column_stack(individual_orbit_sizes(events[:, 0], events[:, 1], mu))
    directions = events[:, 6]
    guide = previous_sizes[selected]
    size_condition = OrbitSizeConstraint(sizes, 10000., 10000.)
    shape = OrbitSizeGuidedConnectivity(ConnectivityShapeConstraint(features, .05), size_condition)
    helpers["make_cluster_validator"] = augmented_validator_factory(helpers["make_cluster_validator"], size_condition)
    observables = np.column_stack((events[:, :2], sizes))
    joint_limits = np.array([.02, .04, 10000., 10000.])
    merge = make_bridge_moment_prefiltered_merge(
        helpers["merge_valid_clusters"], observables, directions, joint_limits, min_size=10,
    )
    recovery = make_moment_prefiltered_recovery(
        helpers["recover_unassigned_members"], observables, directions, joint_limits, min_size=10,
    )
    helpers["merge_valid_clusters"], helpers["recover_unassigned_members"] = merge, recovery
    valid = helpers["make_cluster_validator"](
        events[:, :3], features, limits, 10, 2.5, directions=directions, shape_condition=shape,
    )
    signature = hashlib.sha256()
    for array in (ids, events, features, sizes, limits, joint_limits, np.array([soi])):
        signature.update(np.ascontiguousarray(array).tobytes())
    for source in (Path(__file__), SOURCE, HERE/"ga16_parking_orbit_filter.py",
                   HERE/"ga16_orbit_size_constraints.py", HERE/"ga16_bridge_merge_acceleration.py",
                   HERE/"ga16_size_merge_acceleration.py", HERE/"ga16_size_recovery_acceleration.py",
                   HERE/"ga16_refinement_checkpoint.py"):
        signature.update(source.read_bytes())
    fingerprint = signature.hexdigest()
    result_path = work / "result.npz"
    if result_path.exists():
        with np.load(result_path) as saved:
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError("Il risultato filtrato salvato usa codice o dati diversi.")
            labels, baseline = saved["labels"], saved["baseline"]
        seconds = json.loads((work/"clustering_summary.json").read_text())["clustering_seconds"]
        print("Risultato filtrato riutilizzato.", flush=True)
    else:
        if args.plots_only:
            parser.error("La prova filtrata non e' stata ancora calcolata.")
        tick = time.monotonic()
        lookup = np.full(len(keep), -1, dtype=np.int64)
        lookup[selected] = np.arange(len(selected))
        raw_pool, raw_labels = candidate_pool(CANDIDATES)
        if not np.array_equal(raw_labels, full_catalog[:, 1]):
            raise ValueError("Il pool K83 non corrisponde al catalogo originale.")
        base = base_partition(raw_pool, len(keep))[selected]
        pool = trimmed_pool(raw_pool, lookup)
        del raw_pool
        for old_work, expected in ((PREVIOUS_WORK, previous_eps), (SIZE_WORK, previous_sizes)):
            old_pool, old_labels = candidate_pool(old_work/"candidates.npz")
            if not np.array_equal(old_labels, expected):
                raise ValueError(f"Pool precedente incompatibile: {old_work}")
            pool.extend(trimmed_pool(old_pool, lookup))
            del old_pool

        def proposed_kmeans(features_arg, directions_arg, n_base, random_state=42):
            if features_arg is not features or directions_arg is not directions or n_base != 180:
                raise ValueError("Proposta di partizione richiesta con dati o parametri inattesi.")
            with np.load(PREVIOUS_WORK/f"kmeans_K180_seed{random_state}.npz") as stored:
                if str(stored["fingerprint"]) != previous_fingerprint:
                    raise ValueError("La partizione proposta non corrisponde all'archivio originale.")
                proposed = stored["labels"][selected]
            print(f"Seed {random_state}: proposta precedente ritagliata, rivalidata sulle nuove scale.", flush=True)
            return proposed

        helpers["kmeans_partition"] = proposed_kmeans
        print(f"Filtro: {len(keep):,} -> {len(selected):,} traiettorie; "
              f"solo ellittiche con 0 < a <= {soi:.9f} km. Nessuna integrazione.", flush=True)
        labels, baseline, pool = refine_with_checkpoint(
            helpers, work/"stages", fingerprint,
            [raw_labels[selected], base, previous_eps[selected], guide], events[:, :3], features,
            directions, limits, 10, len(ids)//100, 2.5, seeds=(42, 7, 123), base_count=180,
            candidates=pool, time_limit=30., shape_condition=shape,
        )
        if not all(valid(g) for g in helpers["cluster_groups"](labels)):
            raise RuntimeError("Un cluster filtrato non rispetta i vincoli.")
        seconds = time.monotonic()-tick
        pending = work/"result.tmp.npz"
        np.savez_compressed(pending, labels=labels, baseline=baseline, source_ids=ids,
                            selected_full_rows=selected, std_limits=limits,
                            size_std_limits_km=joint_limits[2:], earth_soi_km=soi, fingerprint=fingerprint)
        pending.replace(result_path)
        (work/"clustering_summary.json").write_text(json.dumps(dict(clustering_seconds=seconds))+"\n")
        helpers["save_fusion_cache"](work/"candidates.npz", fingerprint, pool, labels)
        del pool
    groups = helpers["cluster_groups"](labels)
    if not all(valid(g) for g in groups):
        raise RuntimeError("Il risultato filtrato non supera la verifica finale.")
    output = HERE/"Clusters"/f"GA_16_plots_{TAG}_K{len(groups)}"
    output.mkdir(parents=True, exist_ok=True)
    counts = dict(input=len(keep), retained=len(selected), excluded=int(np.count_nonzero(~keep)),
                  excluded_non_elliptic=int(np.count_nonzero(full_events[:, 0] >= 0)),
                  excluded_elliptic_a_above_soi=int(np.count_nonzero((full_events[:, 0] < 0)&~keep)))
    after = summarize(labels, events, helpers, limits)
    after.update(size_summary(labels, sizes, groups))
    report = dict(filter=counts, earth_soi_km=soi, criterion="elliptic and 0 < a <= Earth SOI",
                  starting_guide_clusters=len(helpers["cluster_groups"](guide)),
                  starting_guide_unassigned=int(np.count_nonzero(guide < 0)), after=after,
                  size_std_limits_km=joint_limits[2:].tolist(), minimum_size=10, max_sigma=2.5,
                  eps_e_connectivity_radius=.05, integration_calls=0, fresh_kmeans_runs=0,
                  proposal_method="Previous partitions trimmed; revalidated on filtered normalization, not refitted KMeans.",
                  clustering_seconds=seconds, fingerprint=fingerprint, output=str(output),
                  pair_prefilters=merge.prefilter_runs, bridge_prefilters=merge.bridge_runs,
                  recovery_prefilters=recovery.prefilter_runs)
    (output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.plots_only:
        helpers["save_cluster_velocity_statistics"](
            output/"cluster_velocity_statistics.txt", labels, events[:, 3:5], full_catalog[selected, 5:7],
            f"CR3BP_prova_GA_16_{TAG}", full_catalog[selected, 7:9], directions,
        )
        catalog = np.lib.format.open_memmap(output/"trajectory_exit_catalog.npy", mode="w+",
                                          dtype=np.float64, shape=(len(selected), 12))
        for first in range(0, len(selected), 100000):
            stop = first+100000
            catalog[first:stop] = full_catalog[selected[first:stop]]
            catalog[first:stop, 1] = labels[first:stop]
        catalog.flush()
        np.savetxt(output/"trajectory_exit_catalog.txt", catalog, fmt=("%d", "%d")+("%.17e",)*10,
                   header="source_id cluster_id vx_soi_km_s vy_soi_km_s vz_soi_km_s t_perigee_days t_soi_days "
                          "x_soi_km y_soi_km eps_perigee_km2_s2 e_perigee omega_perigee_deg\n"
                          f"Source: {ROOT/'CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP_x.txt'}\n"
                          f"Selection: elliptic and 0 < a <= {soi:.17e} km; STD eps < .02, STD e < .04, STD a/rp <= 10000 km.")
        save_orbit_size_dispersion(catalog, output)
    else:
        catalog = np.load(output/"trajectory_exit_catalog.npy", mmap_mode="r")
    if not args.skip_plots:
        complete_figures(catalog, output, [("K2219 ritagliato (da rivalidare)", guide),
                                          ("Split sul campione filtrato", baseline),
                                          ("Fusione finale", labels)], args.workers)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
