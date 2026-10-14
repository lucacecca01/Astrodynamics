"""Recover GA16 plots from the saved exit catalog, without running clustering.

All catalog rows are used for feature plots. Orbit plots reconstruct every
assigned trajectory with the original integration routine and retain the
curves in a resumable disk cache. Existing complete PNGs are left untouched.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE))


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def input_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cached_table(path, cache_dir, name, *, usecols=None, skiprows=0):
    """Avoid reparsing large text files on resume; never modify the source."""
    signature = {"input": input_signature(path), "usecols": usecols, "skiprows": skiprows}
    signature = json.loads(json.dumps(signature))
    binary = cache_dir / f"{name}.npy"
    metadata = cache_dir / f"{name}.json"
    if binary.exists() and metadata.exists() and json.loads(metadata.read_text()) == signature:
        return np.load(binary, mmap_mode="r", allow_pickle=False)
    print(f"Lettura {Path(path).name}...", flush=True)
    values = np.loadtxt(path, usecols=usecols, skiprows=skiprows, ndmin=2)
    temporary = binary.with_name(binary.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(binary)
    atomic_json(metadata, signature)
    del values
    return np.load(binary, mmap_mode="r", allow_pickle=False)


def catalog_source(path):
    with path.open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            if line.startswith("# Source:"):
                source = Path(line.partition(":")[2].strip())
                return source if source.is_absolute() else path.parent / source
    raise ValueError("Il catalogo non specifica il database originale nell'header Source.")


def load_normalization(source_script):
    """Read only the pure normalization helper; never execute the GA16 script."""
    tree = ast.parse(source_script.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "normalize_block")
    weight = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TIME_WEIGHT"
                                               for t in node.targets):
            weight = float(ast.literal_eval(node.value))
    if weight is None:
        raise ValueError("TIME_WEIGHT mancante nello script di riferimento.")
    scope = {"np": np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_script), "exec"), scope)
    return scope["normalize_block"], weight


def prepare_groups(catalog, stats_path, source_script):
    if catalog.ndim != 2 or catalog.shape[1] != 12 or not len(catalog):
        raise ValueError("Atteso il catalogo GA16 completo con 12 colonne.")
    for start in range(0, len(catalog), 100_000):
        block = catalog[start:start + 100_000]
        if not np.isfinite(block).all() or np.any(block[:, :2] != np.floor(block[:, :2])):
            raise ValueError("Catalogo con dati non finiti o identificativi non interi.")
    source_ids = catalog[:, 0].astype(np.int64)
    labels = catalog[:, 1].astype(np.int64)
    if source_ids[0] < 0 or np.any(np.diff(source_ids) <= 0) or np.any(labels < -1):
        raise ValueError("Ordine delle righe o etichette del catalogo non validi.")
    ordered = np.flatnonzero(labels >= 0)
    ordered = ordered[np.argsort(labels[ordered], kind="stable")]
    if not len(ordered):
        raise ValueError("Nessuna traiettoria assegnata.")
    groups = np.split(ordered, np.flatnonzero(np.diff(labels[ordered])) + 1)
    cluster_ids = np.array([labels[g[0]] for g in groups])
    sizes = np.array([len(g) for g in groups])
    stats = np.loadtxt(stats_path, ndmin=2)
    if not np.array_equal(stats[:, :2], np.column_stack((cluster_ids, sizes))):
        raise ValueError("Catalogo e statistiche dei cluster non coincidono.")

    normalize, weight = load_normalization(source_script)
    # Match GA16 exactly: separate global normalization of velocity and time,
    # including unassigned trajectories in both blocks.
    features = np.column_stack((normalize(catalog[:, 2:5]), weight * normalize(catalog[:, 6:7])))
    representatives = np.empty(len(groups), dtype=np.int64)
    for j, members in enumerate(groups):
        values = features[members]
        distance_squared = np.sum((values - values.mean(axis=0))**2, axis=1)
        representatives[j] = members[np.argmin(distance_squared)]
    return source_ids, labels, cluster_ids, sizes, groups, representatives


def expected_plot_names(cluster_ids):
    k = len(cluster_ids)
    pages = (k + 23) // 24
    names = [f"features_2D_K{k}.png", f"features_3D_K{k}.png",
             "representatives.png", "representativesGeocentric_inertial_.png"]
    for prefix in ("", "Geocentric_inertial_"):
        names.extend(f"{prefix}clusters_{page:03d}.png" for page in range(1, pages + 1))
    for cid in cluster_ids:
        names.extend((f"cluster_{cid:03d}_features.png", f"cluster_{cid:03d}_velocity_time.png"))
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "Clusters/GA_16_plots_vinf_time_K798")
    parser.add_argument("--source-script", type=Path, default=HERE / "CR3BP_prova_GA_16_vinf.py")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--stage", choices=("all", "features", "orbits", "verify"), default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers deve essere positivo")
    output = args.output.resolve()
    source_script = args.source_script.resolve()
    cache_dir = (args.cache or WORKSPACE / ".cache/ga16_plot_recovery" / output.name).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output / "trajectory_exit_catalog.txt"
    start = time.monotonic()

    # Separate stage locks allow feature rendering and orbit recovery to run
    # concurrently, but prevent two writers to the same resumable orbit cache.
    import fcntl
    lock_streams = []
    stages = ("features", "orbits") if args.stage == "all" else (args.stage,)
    for stage in stages:
        stream = (cache_dir / f"{stage}.lock").open("w")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Recupero {stage} già in esecuzione per questa cartella.")
        lock_streams.append(stream)

    # Table cache writes are serialized even when the two stages run together.
    with (cache_dir / "tables.lock").open("w") as table_lock:
        fcntl.flock(table_lock, fcntl.LOCK_EX)
        catalog = cached_table(catalog_path, cache_dir, "exit_catalog")
    with threadpool_limits(limits=1):
        source_ids, labels, cluster_ids, sizes, groups, representatives = prepare_groups(
            catalog, output / "cluster_velocity_statistics.txt", source_script,
        )
    print(f"Catalogo: {len(catalog)} traiettorie, {len(groups)} cluster, "
          f"{np.count_nonzero(labels < 0)} non assegnate. Clustering riutilizzato.", flush=True)

    representative_table = np.column_stack((cluster_ids, representatives, source_ids[representatives]))
    temporary = cache_dir / f"representatives_{args.stage}.tmp"
    np.savetxt(temporary, representative_table, fmt="%d",
               header="cluster_id catalog_row source_id; representatives in normalized velocity/time space")
    temporary.replace(cache_dir / "representatives.txt")

    if args.stage in ("all", "features"):
        from ga16_recover_feature_plots import recover_feature_plots
        recover_feature_plots(catalog, cluster_ids, groups, representatives, output,
                              workers=min(args.workers, 4))

    if args.stage in ("all", "orbits"):
        from ga16_recover_orbit_plots import build_orbit_cache, render_orbit_plots
        source_path = catalog_source(catalog_path)
        with (cache_dir / "tables.lock").open("w") as table_lock:
            fcntl.flock(table_lock, fcntl.LOCK_EX)
            all_states = cached_table(source_path, cache_dir, "source_states",
                                      usecols=tuple(range(6)), skiprows=1)
        # The unassigned-orbits figure already exists; every assigned orbit is
        # needed in the inertial panels, even when its rotating panel exists.
        rows = np.flatnonzero(labels >= 0)
        selected_sources = source_ids[rows]
        if selected_sources[-1] >= len(all_states):
            raise ValueError("source_id fuori dal database originale.")
        states = all_states[selected_sources]
        representative_sources = source_ids[representatives]
        representative_indices = np.searchsorted(selected_sources, representative_sources)
        preview_saved = False

        def save_ready_representatives(cache):
            nonlocal preview_saved
            if not preview_saved and np.all(cache.done[representative_indices]):
                render_orbit_plots(cache, output, labels[rows], cluster_ids, sizes,
                                   representative_sources, ready_only=True)
                preview_saved = True

        orbit_cache = build_orbit_cache(
            source_script, cache_dir / "orbits", selected_sources, states,
            workers=args.workers, batch_size=2048,
            expected_times_days=catalog[rows, 5:7],
            expected_soi_positions_km=catalog[rows, 7:9],
            priority_source_ids=representative_sources,
            on_batch=save_ready_representatives,
        )
        del states, all_states
        render_orbit_plots(orbit_cache, output, labels[rows], cluster_ids, sizes,
                           representative_sources)

    from PIL import Image
    missing = []
    for name in expected_plot_names(cluster_ids):
        try:
            with Image.open(output / name) as img:
                img.verify()
        except (OSError, SyntaxError):
            missing.append(name)
    report = {"stage": args.stage, "catalog": input_signature(catalog_path),
              "source_script_sha256": hashlib.sha256(source_script.read_bytes()).hexdigest(),
              "n_trajectories": len(catalog), "n_clusters": len(cluster_ids),
              "expected_recovery_plots": len(expected_plot_names(cluster_ids)),
              "missing_or_invalid": missing, "elapsed_seconds": time.monotonic() - start}
    atomic_json(cache_dir / f"report_{args.stage}.json", report)
    print(f"Fase {args.stage} terminata in {report['elapsed_seconds']:.1f} s; "
          f"grafici ancora mancanti: {len(missing)}.", flush=True)
    if args.stage in ("all", "verify") and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
