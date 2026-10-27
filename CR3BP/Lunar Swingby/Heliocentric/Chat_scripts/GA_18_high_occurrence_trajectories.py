"""Plot every unique GA16 trajectory in cells with occurrences >= a threshold.

Selections are separate for maps 05--10 and use the winning transfers in the
saved GA18 report. Cached CR3BP curves are checked against the CURRENT catalog;
there is no Lambert calculation and no trajectory subsampling.
"""

import argparse
import hashlib
import inspect
from multiprocessing import get_context
import os
import time
import json
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

import GA_18_plots as plots
from Celestial_Mechanics import Integration

_CR3BP_MODEL = Integration()

WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = WORKSPACE / ".cache/ga16_plot_recovery/GA_16_plots_vinf_time_K798/orbits"
TITLES = {
    "05": r"05: $\varepsilon$–$e$",
    "06": r"06: $\varepsilon$–$\omega$",
    "07": r"07: $h_p$–$e$",
    "08": r"08: $h_p$–$\omega$",
    "09": r"09: $h_p$–$\theta$",
    "10": r"10: $\varepsilon$–$\theta$",
}


def select_members(members, stats, winner_rows, catalog_path, threshold=None,
                   only_bound=True, n_bins=plots.N_BINS, mode="asteroids", fraction=0.9):
    """Return exact cell selections, keeping the original histogram bin edges."""
    result = {}
    winning_ids = stats[np.unique(winner_rows), 0]
    eps, ecc = members[:, 9], members[:, 10]
    keep = members[:, 1] >= 0
    if only_bound:
        bound = (eps < 0) & (ecc >= 0) & (ecc < 1)
        rp = np.full(len(members), np.nan)
        rp[bound] = -plots.MU_EARTH / (2 * eps[bound]) * (1 - ecc[bound])
        keep &= bound & (rp > plots.R_EARTH_KM)
    admitted = members[keep]
    winning = np.isin(admitted[:, 1], winning_ids)
    for key, variable in (("05", "e"), ("06", "omega")):
        geometry = plots.prepare_parking_geometry(members, stats, only_bound, n_bins, variable)
        data = plots.best_transfer_occurrences(geometry, winner_rows, len(stats), mode)
        x = admitted[:, 9]
        y = admitted[:, 10] if variable == "e" else admitted[:, 11] % 360
        ix = np.clip(np.searchsorted(data["edges_x"], x, side="right") - 1, 0, n_bins - 1)
        iy = np.clip(np.searchsorted(data["edges_y"], y, side="right") - 1, 0, n_bins - 1)
        cutoff = (max(1, int(np.ceil(fraction * data["occurrences"].max())))
                  if threshold is None else threshold)
        data["selection_threshold"] = cutoff
        selected = winning & (data["occurrences"][ix, iy] >= cutoff)
        data["selected_source_ids"] = admitted[selected, 0].astype(np.int64)
        result[key] = data
    for data, filename, *_ in plots.additional_best_transfer_maps(
            members, stats, winner_rows, catalog_path, only_bound, n_bins, mode,
            occurrence_threshold=threshold, occurrence_fraction=fraction if threshold is None else None):
        result[filename[:2]] = data
    return result


def checked_curves(cache_directory, members, allow_missing=False):
    """Read only selected curves, verifying IDs, events, elements and SOI exit."""
    directory = Path(cache_directory)
    source_ids = members[:, 0].astype(np.int64)
    cached_ids = np.load(directory / "source_ids.npy", mmap_mode="r", allow_pickle=False)
    indices = np.searchsorted(cached_ids, source_ids)
    available = (indices < len(cached_ids)) & (cached_ids[np.minimum(indices, len(cached_ids)-1)] == source_ids)
    if not available.all() and allow_missing:
        n_plot = json.loads((directory / "metadata.json").read_text())["n_plot"]
        synodic = np.full((len(members), 2, 2*n_plot-1), np.nan)
        inertial = np.full_like(synodic, np.nan)
        directions = np.full(len(members), np.nan)
        if available.any():
            synodic[available], inertial[available], directions[available], _ = checked_curves(
                directory, members[available])
        print(f"Cache originale: {available.sum()}/{len(members)} traiettorie disponibili; "
              f"{(~available).sum()} ricostruite direttamente dal database.", flush=True)
        return synodic, inertial, directions, n_plot
    if not available.all():
        raise ValueError("Traiettorie selezionate assenti dalla cache indicata.")
    done = np.load(directory / "done.npy", mmap_mode="r", allow_pickle=False)
    if not np.all(done[indices]):
        raise ValueError("La cache contiene traiettorie selezionate ancora incomplete.")
    curves = np.load(directory / "curves.npy", mmap_mode="r", allow_pickle=False)[indices]
    bounds = np.load(directory / "bounds.npy", mmap_mode="r", allow_pickle=False)[indices]
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)[indices]
    if (not np.isfinite(curves).all() or not np.isfinite(features).all()
            or not np.allclose(bounds * plots.TIME_UNIT_S / 86400, members[:, 5:7],
                               rtol=1e-9, atol=1e-8)
            or not np.allclose(features[:, :2], members[:, 9:11], rtol=1e-8, atol=1e-8)
            or np.any(np.abs((features[:, 2] - members[:, 11] + 180) % 360 - 180) > 1e-8)):
        raise ValueError("Cache non coerente con eventi/elementi del catalogo corrente.")
    metadata = json.loads((directory / "metadata.json").read_text())
    n_plot = metadata["n_plot"]
    if curves.shape[1:] != (2, 2 * n_plot - 1):
        raise ValueError("Campionamento della cache incoerente.")
    mu = plots.MOON_MASS_KG / (5.974e24 + plots.MOON_MASS_KG)
    synodic = curves.copy()
    synodic[:, 0] += mu
    synodic *= plots.EARTH_MOON_DISTANCE_KM
    times = np.concatenate((np.linspace(bounds[:, 0], 0, n_plot, axis=1),
                            np.linspace(0, bounds[:, 1], n_plot, axis=1)[:, 1:]), axis=1)
    cosine, sine = np.cos(times), np.sin(times)
    inertial = np.empty_like(synodic)
    inertial[:, 0] = cosine * synodic[:, 0] - sine * synodic[:, 1]
    inertial[:, 1] = sine * synodic[:, 0] + cosine * synodic[:, 1]
    if not np.allclose(inertial[:, :, -1], members[:, 7:9], rtol=1e-9, atol=1e-5):
        raise ValueError("Posizioni SOI in cache diverse dal catalogo corrente.")
    return synodic, inertial, features[:, 6], n_plot


def cr3bp_rhs(t, state):
    """Same nondimensional CR3BP equations and masses as GA16."""
    mu = plots.MOON_MASS_KG / (5.974e24 + plots.MOON_MASS_KG)
    return _CR3BP_MODEL.CR3BP_ODE(t, state, mu)


def _integrate_dense(task):
    from scipy.integrate import solve_ivp
    index, state, peri, soi, n_plot = task
    # Evaluate dense DOP853 solutions at NEW times, rather than interpolating
    # the old 99-point polylines. Time zero is retained exactly once.
    # Match the original GA16 RHS and tolerances. Integrate past the saved
    # endpoints so these remain dense-output evaluations, as in the catalog,
    # rather than changing the final adaptive step at the selected event.
    backward = solve_ivp(cr3bp_rhs, (0, max(-4*np.pi, peri-1)), state, method="DOP853",
                         rtol=1e-9, atol=1e-12, t_eval=np.linspace(0, peri, n_plot))
    forward = solve_ivp(cr3bp_rhs, (0, min(4*np.pi, soi+1)), state, method="DOP853",
                        rtol=1e-9, atol=1e-12, t_eval=np.linspace(0, soi, n_plot))
    if not backward.success or not forward.success:
        raise ValueError(f"Integrazione densa fallita, indice {index}.")
    mu = plots.MOON_MASS_KG / (5.974e24 + plots.MOON_MASS_KG)
    r = backward.y[:3, -1] + [mu, 0, 0]
    v = backward.y[3:, -1] + np.cross([0, 0, 1], r)
    h = np.cross(r, v)
    direction = np.sign(h[2])
    energy = (v @ v / 2 - (1-mu)/np.linalg.norm(r)) * (
        plots.MU_EARTH + 6.67430e-20*plots.MOON_MASS_KG) / plots.EARTH_MOON_DISTANCE_KM
    ev = np.cross(v, h)/(1-mu) - r/np.linalg.norm(r)
    omega = direction * np.rad2deg(np.arctan2(ev[1], ev[0]) + peri) % 360
    features = np.array([energy, np.linalg.norm(ev), omega, direction])
    return index, np.concatenate((backward.y[:2, ::-1], forward.y[:2, 1:]), axis=1), features


def dense_curves(cache_directory, members, sparse_synodic, n_plot, workers, return_inertial=True):
    """Resume true reintegration of every selected orbit at >= twice the samples."""
    ids = members[:, 0].astype(np.int64)
    source_path = Path(cache_directory).parent / "source_states.npy"
    all_states = np.load(source_path, mmap_mode="r", allow_pickle=False)
    states = all_states[ids]
    mu = plots.MOON_MASS_KG / (5.974e24 + plots.MOON_MASS_KG)
    etd_positions = states[:, :2].copy()
    etd_positions[:, 0] += mu
    etd_positions *= plots.EARTH_MOON_DISTANCE_KM
    available = np.isfinite(sparse_synodic).all(axis=(1, 2))
    if not np.allclose(etd_positions[available], sparse_synodic[available, :, sparse_synodic.shape[2] // 2],
                       rtol=0, atol=1e-7):
        raise ValueError("Stati ETD nella cache diversi dalle traiettorie originali.")
    bounds = members[:, 5:7] * 86400 / plots.TIME_UNIT_S
    signature = dict(n_plot=n_plot, rtol=1e-9, atol=1e-12,
                     source=str(source_path.resolve()), mtime_ns=source_path.stat().st_mtime_ns,
                     rhs=hashlib.sha256((inspect.getsource(cr3bp_rhs) + inspect.getsource(Integration.CR3BP_ODE)
                                         + inspect.getsource(_integrate_dense)).encode()).hexdigest())
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(members).tobytes())
    directory = WORKSPACE / ".cache/ga18_high_occurrence_dense" / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    path, done_path = directory / "curves.npy", directory / "done.npy"
    shape = (len(ids), 2, 2*n_plot-1)
    if path.exists():
        curves = np.load(path, mmap_mode="r+", allow_pickle=False)
        done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
        fresh_features = np.load(directory / "features.npy", mmap_mode="r+", allow_pickle=False)
        if curves.shape != shape or done.shape != (len(ids),):
            raise ValueError("Forma della cache densa incoerente.")
    else:
        curves = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(len(ids),))
        fresh_features = np.lib.format.open_memmap(directory / "features.npy", mode="w+", dtype=np.float64,
                                                   shape=(len(ids), 4))
        done[:] = False
        done.flush()
        np.save(directory / "source_ids.npy", ids)
        (directory / "metadata.json").write_text(json.dumps(signature, indent=2)+"\n")
    # A snapshot lets selections share validated curves even if a figure run
    # was interrupted before its final report was written.
    snapshot_path = directory / "members.npy"
    if not snapshot_path.exists():
        np.save(snapshot_path, members)
    for previous_snapshot in directory.parent.glob("*/members.npy"):
        previous_dir = previous_snapshot.parent
        if previous_dir == directory or json.loads((previous_dir / "metadata.json").read_text()) != signature:
            continue
        previous_members = np.load(previous_snapshot, mmap_mode="r")
        previous_ids = previous_members[:, 0].astype(np.int64)
        _, current_indices, previous_indices = np.intersect1d(ids, previous_ids, return_indices=True)
        new = ~done[current_indices]
        current_indices, previous_indices = current_indices[new], previous_indices[new]
        if not np.array_equal(members[current_indices], previous_members[previous_indices]):
            continue
        previous_done = np.load(previous_dir / "done.npy", mmap_mode="r")
        ready = previous_done[previous_indices]
        current_indices, previous_indices = current_indices[ready], previous_indices[ready]
        previous_curves = np.load(previous_dir / "curves.npy", mmap_mode="r")
        previous_features = np.load(previous_dir / "features.npy", mmap_mode="r")
        for start in range(0, len(current_indices), 4096):
            current = current_indices[start:start+4096]
            previous = previous_indices[start:start+4096]
            curves[current] = previous_curves[previous]
            fresh_features[current] = previous_features[previous]
            done[current] = True
        if len(current_indices):
            print(f"Recuperate {len(current_indices)} curve dense da una selezione precedente.", flush=True)
    # Reuse previously validated dense trajectories from earlier figure sets.
    for report_path in plots.OUTPUT_DIR.glob("traiettorie_*/riepilogo.json"):
        report = json.loads(report_path.read_text())
        previous_dir = Path(report.get("dense_cache", ""))
        member_path = report_path.parent / "membri_selezionati.npz"
        if previous_dir == directory or not (previous_dir / "metadata.json").is_file() or not member_path.is_file():
            continue
        if json.loads((previous_dir / "metadata.json").read_text()) != signature:
            continue
        with np.load(member_path, allow_pickle=False) as previous_data:
            previous_members = previous_data["members"]
        previous_ids = previous_members[:, 0].astype(np.int64)
        _, current_indices, previous_indices = np.intersect1d(ids, previous_ids, return_indices=True)
        new = ~done[current_indices]
        current_indices, previous_indices = current_indices[new], previous_indices[new]
        if not np.array_equal(members[current_indices], previous_members[previous_indices]):
            continue
        previous_done = np.load(previous_dir / "done.npy", mmap_mode="r")
        ready = previous_done[previous_indices]
        current_indices, previous_indices = current_indices[ready], previous_indices[ready]
        previous_curves = np.load(previous_dir / "curves.npy", mmap_mode="r")
        previous_features = np.load(previous_dir / "features.npy", mmap_mode="r")
        for start in range(0, len(current_indices), 4096):
            current = current_indices[start:start+4096]
            previous = previous_indices[start:start+4096]
            curves[current] = previous_curves[previous]
            fresh_features[current] = previous_features[previous]
            done[current] = True
        if len(current_indices):
            print(f"Riutilizzate {len(current_indices)} traiettorie dense già verificate.", flush=True)
    pending = np.flatnonzero(~done)
    started = last_report = time.monotonic()
    print(f"Reintegrazione densa: {done.sum()}/{len(ids)} pronte; "
          f"{2*n_plot-1} punti per traiettoria, {workers} processi.", flush=True)
    tasks = ((int(i), states[i], *bounds[i], n_plot) for i in pending)
    with get_context("fork").Pool(workers) as pool:
        for i, curve, features in pool.imap_unordered(_integrate_dense, tasks, chunksize=8):
            synodic = curve.copy()
            synodic[0] += mu
            synodic *= plots.EARTH_MOON_DISTANCE_KM
            # Compare both ends with the independently validated original cache.
            if available[i]:
                endpoint_error = np.linalg.norm(synodic[:, [0, -1]] - sparse_synodic[i][:, [0, -1]], axis=0)
                if endpoint_error.max() > 1.0:
                    raise ValueError(f"source_id {ids[i]}: scarto estremi {endpoint_error} km.")
            c, sn = np.cos(bounds[i, 1]), np.sin(bounds[i, 1])
            soi_position = np.array([[c, -sn], [sn, c]]) @ synodic[:, -1]
            angle_error = (features[2] - members[i, 11] + 180) % 360 - 180
            if (not np.isfinite(synodic).all()
                    or np.linalg.norm(soi_position-members[i, 7:9]) > 1.0
                    or not np.allclose(features[:2], members[i, 9:11], rtol=1e-5, atol=1e-6)
                    or abs(angle_error) > 0.02):
                raise ValueError(f"source_id {ids[i]}: reintegrazione incoerente col catalogo.")
            curves[i] = synodic
            fresh_features[i] = features
            done[i] = True
            now = time.monotonic()
            if now-last_report > 25:
                curves.flush()
                fresh_features.flush()
                done.flush()
                count = int(done.sum())
                rate = (count - (len(ids)-len(pending))) / (now-started)
                print(f"Dense {count}/{len(ids)}; {rate:.1f} traiettorie/s", flush=True)
                last_report = now
    curves.flush()
    fresh_features.flush()
    done.flush()
    if not done.all():
        raise ValueError("Reintegrazione incompleta.")
    if not return_inertial:
        return curves, None, fresh_features[:, 3], directory
    times = np.concatenate((np.linspace(bounds[:, 0], 0, n_plot, axis=1),
                            np.linspace(0, bounds[:, 1], n_plot, axis=1)[:, 1:]), axis=1)
    inertial = np.empty_like(curves)
    cosine, sine = np.cos(times), np.sin(times)
    inertial[:, 0] = cosine*curves[:, 0] - sine*curves[:, 1]
    inertial[:, 1] = sine*curves[:, 0] + cosine*curves[:, 1]
    return curves, inertial, fresh_features[:, 3], directory


def draw_orbits(ax, curves, color_indices, *, zoom=False, inertial=False):
    """Draw one polyline for EVERY input trajectory; coordinates are km."""
    d = plots.EARTH_MOON_DISTANCE_KM
    ax.set_aspect("equal")
    ax.set_xlabel("x geocentrica [km]")
    ax.set_ylabel("y geocentrica [km]")
    ax.grid(alpha=0.15)
    ax.add_patch(Circle((0, 0), plots.R_EARTH_KM, color="#2563eb", zorder=4))
    ax.annotate("Terra", (0, 0), xytext=(5, 6), textcoords="offset points", fontsize=8)
    if inertial:
        ax.add_patch(Circle((0, 0), d, fill=False, ls="--", lw=0.7, color="#999999"))
    else:
        ax.add_patch(Circle((d, 0), 1737, color="#64748b", zorder=4))
        ax.annotate("Luna", (d, 0), xytext=(5, 6), textcoords="offset points", fontsize=8)
        moon_soi = (plots.MOON_MASS_KG / 5.974e24)**0.4 * d
        ax.add_patch(Circle((d, 0), moon_soi, fill=False, ls="--", lw=0.7, color="#999999"))
    if len(curves):
        # Stable cluster colors across ALL maps and both reference frames.
        colors = plt.get_cmap("turbo")(color_indices)
        colors[:, 3] = 0.12
        ax.add_collection(LineCollection(curves.transpose(0, 2, 1), colors=colors,
                                         linewidths=0.35, rasterized=True))
        if zoom:
            ax.set_xlim(d - 100000, d + 100000)
            ax.set_ylim(-100000, 100000)
        else:
            points = curves.transpose(0, 2, 1).reshape(-1, 2)
            low, high = points.min(axis=0), points.max(axis=0)
            low = np.minimum(low, [-plots.R_EARTH_KM, -plots.R_EARTH_KM])
            high = np.maximum(high, [d + 1737, plots.R_EARTH_KM])
            side = max(high - low) * 1.08
            center = (low + high) / 2
            ax.set_xlim(center[0] - side / 2, center[0] + side / 2)
            ax.set_ylim(center[1] - side / 2, center[1] + side / 2)
    else:
        ax.set_xlim(-200000, 1000000)
        ax.set_ylim(-600000, 600000)
        ax.text(0.5, 0.68, "Nessuna cella raggiunge la soglia", transform=ax.transAxes,
                ha="center", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=None,
                        help="Soglia assoluta facoltativa; default: ultimo 10%% della scala di ogni mappa")
    parser.add_argument("--top-percent", type=float, default=10)
    parser.add_argument("--points-per-segment", type=int, default=101,
                        help="Punti per ciascun tratto perigeo–ETD e ETD–SOI; default 201 punti totali")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2)-2)))
    parser.add_argument("--report", type=Path, default=plots.OUTPUT_DIR / "riepilogo.txt")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threshold is not None and args.threshold < 1:
        parser.error("--threshold deve essere positivo")
    if not 0 < args.top_percent <= 100:
        parser.error("--top-percent deve essere in (0, 100]")
    if args.points_per_segment < 101 or args.workers < 1:
        parser.error("Servono almeno 101 punti per tratto e almeno un processo")
    fraction = 1 - args.top_percent / 100
    report = args.report.read_text()
    catalog_stats = Path(re.search(r"^Catalogo: (.+)$", report, re.M)[1])
    catalog_members = catalog_stats.with_name("trajectory_exit_catalog.txt")
    stats = plots.load_table(catalog_stats, 14)
    stats = stats[np.argsort(stats[:, 0])]
    members = plots.load_table(catalog_members, 12)
    winner_ids = np.array([int(x) for x in re.findall(r"Miglior trasferimento: cluster (\d+)", report)])
    winner_rows = np.searchsorted(stats[:, 0], winner_ids)
    if not np.array_equal(stats[winner_rows, 0], winner_ids):
        raise ValueError("Cluster vincenti del report assenti dal catalogo.")
    only_bound = "Orbita di parcheggio: legata, perigeo sopra la superficie" in report
    mode = re.search(r"Occorrenze migliori trasferimenti: (\w+);", report)[1]
    selections = select_members(members, stats, winner_rows, catalog_members,
                                args.threshold, only_bound, mode=mode, fraction=fraction)
    ids = np.unique(np.concatenate([d["selected_source_ids"] for d in selections.values()]))
    selected_members = members[np.isin(members[:, 0], ids)]
    selected_members = selected_members[np.argsort(selected_members[:, 0])]
    del members
    if len(selected_members):
        sparse_synodic, _, directions, old_n_plot = checked_curves(args.cache, selected_members, allow_missing=True)
        n_plot = max(args.points_per_segment, 2*old_n_plot)
        synodic, inertial, directions, dense_cache = dense_curves(
            args.cache, selected_members, sparse_synodic, n_plot, args.workers)
        del sparse_synodic
    else:
        n_plot = args.points_per_segment
        synodic = inertial = np.empty((0, 2, 2*n_plot-1))
        directions, dense_cache = np.empty(0), None
    cluster_ids = np.unique(selected_members[:, 1])
    colors = np.searchsorted(cluster_ids, selected_members[:, 1]) / max(1, len(cluster_ids) - 1)
    policy = (f"occorrenze_ge_{args.threshold}" if args.threshold is not None
              else f"top_{args.top_percent:g}_percent")
    output = args.output or args.report.parent / f"traiettorie_{policy}_{2*n_plot-1}_punti"
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(threshold=args.threshold,
                   top_percent_of_each_map_scale=args.top_percent if args.threshold is None else None,
                   dense_cache=str(dense_cache), mode=mode, n_bins=plots.N_BINS,
                   report=str(args.report.resolve()), catalog=str(catalog_members.resolve()),
                   cache=str(args.cache.resolve()), unique_union=len(ids),
                   samples_per_trajectory=2 * n_plot - 1, maps={})
    overview, overview_axes = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    for (key, data), overview_ax in zip(selections.items(), overview_axes.ravel()):
        selected_ids = np.sort(data["selected_source_ids"])
        indices = np.searchsorted(ids, selected_ids)
        threshold = data["selection_threshold"]
        n_cells = int((data["occurrences"] >= threshold).sum())
        description = f"{len(indices)} traiettorie distinte — {n_cells} celle"
        summary["maps"][key] = dict(threshold=threshold, trajectories=len(indices), cells=n_cells,
                                    maximum_occurrences=int(data["occurrences"].max()))
        np.savetxt(output / f"{key}_traiettorie_selezionate.txt",
                   selected_members[indices][:, :2], fmt="%d", header="source_id cluster_id")
        np.savez_compressed(output / f"{key}_selezione_celle.npz", source_ids=selected_ids,
                            occurrences=data["occurrences"], edges_x=data["edges_x"], edges_y=data["edges_y"])
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained")
        for ax, curves, title, zoom, inertial_frame in (
            (axes[0], synodic[indices], "Sinodico Terra–Luna", False, False),
            (axes[1], synodic[indices], "Sinodico: dettaglio presso la Luna", True, False),
            (axes[2], inertial[indices], "Geocentrico inerziale: Luna su +x a ETD", False, True),
        ):
            draw_orbits(ax, curves, colors[indices], zoom=zoom, inertial=inertial_frame)
            ax.set_title(title, fontsize=11)
        fig.suptitle(f"{TITLES[key]} — occorrenze ≥ {threshold} (massimo {data['occurrences'].max()})\n{description}", fontsize=15)
        fig.supxlabel("Tutte le traiettorie selezionate, dal perigeo salvato all’uscita dalla SOI terrestre.\n"
                      f"Colori per cluster, identici fra mappe; {2 * n_plot - 1} punti per traiettoria da nuova integrazione CR3BP.", fontsize=10)
        fig.savefig(output / f"{key}_traiettorie_{policy}.png", dpi=180)
        plt.close(fig)
        draw_orbits(overview_ax, synodic[indices], colors[indices])
        overview_ax.set_title(f"{TITLES[key]} — occorrenze ≥ {threshold}\n{description}", fontsize=12)
        print(f"{key}: {description}; soglia {threshold}; massimo {data['occurrences'].max()}", flush=True)
    selection_label = (f"occorrenze ≥ {args.threshold}" if args.threshold is not None
                       else f"ultimo {args.top_percent:g}% della scala di ciascuna mappa")
    overview.suptitle(f"Traiettorie: {selection_label} — {2*n_plot-1} punti per traiettoria", fontsize=16)
    overview.supxlabel("Ogni traiettoria compare una volta per mappa; colori per cluster. Le occorrenze contano gli asteroidi, non le traiettorie.", fontsize=11)
    overview.savefig(output / "00_confronto_sinodico.png", dpi=160)
    plt.close(overview)
    np.savez_compressed(output / "membri_selezionati.npz", members=selected_members,
                        directions=directions, source_ids=ids)
    (output / "riepilogo.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Salvato tutto in {output}; unione: {len(ids)} traiettorie distinte.", flush=True)


if __name__ == "__main__":
    main()
