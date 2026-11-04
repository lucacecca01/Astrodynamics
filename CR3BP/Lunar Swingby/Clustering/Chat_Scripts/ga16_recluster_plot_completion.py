"""Complete reclustering figures using validated, already integrated orbits.

The orbit archive is read in place: no integration, clustering, or multi-GB
copy is performed. Existing valid figures are preserved.
"""

import argparse
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = HERE / "CR3BP_prova_GA_16_eps_e.py"
PREVIOUS = HERE / "Clusters/GA_16_plots_eps_e_K83"
ARCHIVE = ROOT / ".cache/ga16_plot_recovery/GA_16_plots_eps_e_K83"
_RENDER_CONTEXT = None


def open_validated_orbits(directory, catalog, selected_rows, source_script):
    """Check every reused row against its saved events, then open read-only."""
    from ga16_recover_orbit_plots import OrbitCache, _array_digest, load_original_context

    directory = Path(directory)
    context = load_original_context(source_script)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["source_fingerprint"] != context["recovery_fingerprint"]:
        raise ValueError("L'archivio delle orbite usa una dinamica diversa.")
    arrays = {key: np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("source_ids", "curves", "bounds", "features", "done")}
    ids = arrays["source_ids"]
    if (not np.array_equal(ids, catalog[selected_rows, 0])
            or _array_digest(ids) != metadata["source_ids_sha256"]
            or not arrays["done"].all()):
        raise ValueError("Le orbite pronte non corrispondono ai membri assegnati.")
    count = len(ids)
    if (arrays["curves"].shape != (count, 2, 2*context["N_PLOT"]-1)
            or arrays["bounds"].shape != (count, 2)
            or arrays["features"].shape != (count, 7)):
        raise ValueError("Dimensioni dell'archivio delle orbite non valide.")
    for first in range(0, count, 32768):
        last = min(first+32768, count)
        expected = catalog[selected_rows[first:last]]
        features = arrays["features"][first:last]
        bounds = arrays["bounds"][first:last]
        curves = arrays["curves"][first:last]
        days = (bounds-context["T_min"])*context["TU"]/context["day"]
        x, y = curves[:, 0, -1]+context["mu"], curves[:, 1, -1]
        cosine, sine = np.cos(bounds[:, 1]), np.sin(bounds[:, 1])
        positions = context["d"]*np.column_stack((cosine*x-sine*y, sine*x+cosine*y))
        if (not np.isfinite(curves).all()
                or not np.allclose(days, expected[:, 5:7], rtol=1e-9, atol=1e-8)
                or not np.allclose(features[:, :2], expected[:, 9:11], rtol=1e-9, atol=1e-8)
                or not np.allclose(features[:, 3:6], expected[:, 2:5], rtol=1e-9, atol=1e-8)
                or not np.allclose((features[:, 2]-expected[:, 11]+180)%360-180, 0, atol=1e-7)
                or not np.allclose(positions, expected[:, 7:9], rtol=1e-9, atol=1e-4)):
            raise ValueError(f"Eventi dell'archivio incompatibili alle righe {first}:{last}.")
    print(f"Orbite validate e riutilizzate: {count:,}; nuove integrazioni: 0.", flush=True)
    return OrbitCache(directory, context=context, **arrays)


def saved_stages(catalog, work_directory):
    """Reconstruct only the partitions actually retained by the experiment."""
    original = np.load(ARCHIVE / "exit_catalog.npy", mmap_mode="r")
    if original.shape != catalog.shape:
        raise ValueError("Il catalogo originale ha dimensioni diverse.")
    for first in range(0, len(catalog), 100000):
        last = first+100000
        if (not np.array_equal(original[first:last, 0], catalog[first:last, 0])
                or not np.array_equal(original[first:last, 2:], catalog[first:last, 2:])):
            raise ValueError("La prova ha cambiato dati fisici o identificativi delle traiettorie.")
    old_labels = original[:, 1].astype(np.int64)
    with np.load(Path(work_directory) / "result.npz") as result:
        if (not np.array_equal(result["labels"], catalog[:, 1])
                or not np.array_equal(result["source_ids"], catalog[:, 0])):
            raise ValueError("Le etichette salvate non corrispondono al nuovo catalogo.")
        baseline = result["baseline"]
        final = result["labels"]
    with np.load(HERE / "Clusters/GA_16_eps_e_candidates.npz") as pool:
        if not np.array_equal(pool["labels"], old_labels):
            raise ValueError("Il pool originale e il catalogo K83 non corrispondono.")
        pool_members, offsets = pool["members"], pool["offsets"]
        base = np.full(len(catalog), -1, dtype=np.int64)
        covered = 0
        for number, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            members = pool_members[a:b]
            if np.any(base[members] >= 0):
                break
            base[members] = number
            covered += len(members)
            if covered == len(catalog):
                break
    stages = [("K83 precedente (limiti 0.04 / 0.05)", old_labels)]
    if covered == len(catalog):
        stages.append(("Partizione iniziale riutilizzata", base))
    stages.extend((("Split con i nuovi limiti", baseline), ("Fusione finale", final)))
    return stages


def _render_pages(task):
    from ga16_recover_orbit_plots import render_orbit_plots

    cache, output, labels, ids, sizes, representatives = _RENDER_CONTEXT
    inertial, pages = task
    paths = render_orbit_plots(
        cache, output, labels, ids, sizes, representatives,
        render_representatives=False, frame_filter={inertial}, page_filter=set(pages),
    )
    print("Pagine completate: " + ", ".join(path.name for path in paths), flush=True)
    return [str(path) for path in paths]


def complete_plots(output_directory, work_directory, *, workers=3, orbits_only=False):
    from GA_16_recover_plots import expected_plot_names, prepare_groups
    from ga16_recover_feature_plots import valid_png
    from ga16_recover_orbit_plots import render_orbit_plots

    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers deve essere un intero positivo")
    output = Path(output_directory).resolve()
    catalog = np.load(output / "trajectory_exit_catalog.npy", mmap_mode="r")
    with threadpool_limits(limits=1):
        ids, labels, group_ids, sizes, groups, representatives = prepare_groups(
            catalog, output / "cluster_velocity_statistics.txt", SOURCE,
        )
        stages = saved_stages(catalog, work_directory)
        old_labels = stages[0][1]
        if not np.array_equal(labels < 0, old_labels < 0):
            raise ValueError("I non assegnati sono cambiati: il relativo plot va rigenerato.")
        path = output / "unassigned_trajectories.png"
        if not valid_png(path):
            previous = PREVIOUS / path.name
            if not valid_png(previous):
                raise ValueError("Manca il grafico originale dei 19 non assegnati.")
            temporary = path.with_suffix(".png.tmp")
            shutil.copyfile(previous, temporary)
            os.replace(temporary, path)
        selected = np.flatnonzero(labels >= 0)
        cache = open_validated_orbits(ARCHIVE / "orbits", catalog, selected, SOURCE)
        rep_sources = ids[representatives]
        # Draw the two small representative figures before the large pages.
        render_orbit_plots(cache, output, labels[selected], group_ids, sizes,
                           rep_sources, render_clusters=False)
        global _RENDER_CONTEXT
        _RENDER_CONTEXT = (cache, output, labels[selected], group_ids, sizes, rep_sources)
        pages = (len(group_ids)+23)//24
        lanes = max(1, min(workers, pages))
        tasks = [(frame, list(range(lane, pages, lanes)))
                 for frame in (False, True) for lane in range(lanes)]
        if workers == 1:
            for task in tasks:
                _render_pages(task)
        else:
            with get_context("fork").Pool(workers) as pool:
                list(pool.imap_unordered(_render_pages, tasks))
        _RENDER_CONTEXT = None
        if not orbits_only:
            import fcntl
            from ga16_recluster_diagnostics import recover_diagnostics
            with (output / ".diagnostics.lock").open("w") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                recover_diagnostics(catalog, stages, output)
    extra = ["unassigned_trajectories.png", "STD_vs_K.png", "gaussian_shape_vs_stage.png",
             "physical_STD_and_3sigma_vs_K.png", f"cluster_diagnostics_K{len(group_ids)}.png"]
    names = expected_plot_names(group_ids)+extra
    missing = [name for name in names if not valid_png(output/name)]
    report = dict(expected_pngs=len(names), missing_or_invalid=missing,
                  assigned_orbits_reused=int(len(selected)), integration_calls=0,
                  cluster_count=int(len(group_ids)))
    temporary = output / "plot_completion.json.tmp"
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(output / "plot_completion.json")
    print(json.dumps(report, indent=2), flush=True)
    if missing and not orbits_only:
        raise RuntimeError("Alcuni plot sono ancora mancanti o non validi.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "Clusters/GA_16_plots_eps0p02_e0p04_K196")
    parser.add_argument("--work", type=Path, default=ROOT / ".cache/ga16_recluster_eps0p02_e0p04")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--orbits-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers deve essere positivo")
    start = time.monotonic()
    complete_plots(args.output, args.work, workers=args.workers, orbits_only=args.orbits_only)
    print(f"Completamento plot: {time.monotonic()-start:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
