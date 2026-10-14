"""Resume GA16 orbit rendering without running its integration/clustering main.

The caller supplies retained source states and the saved final catalog.  Orbit
samples are written to resumable NumPy files; figures use collections instead
of one Matplotlib artist per trajectory.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from Celestial_Mechanics import Integration, Transformations


_FUNCTIONS = {
    "earth_SOI_exit", "collision", "earth_perigee", "orbital_parameters",
    "first_perigee_parameters", "integrate_one", "integrate_and_sample_one",
    "geocentric_inertial_trajectory", "plot_frame_bodies",
}
_CONSTANTS = {
    "G", "d", "m1", "m2", "ms", "d_E", "R_E", "R_L", "day", "M",
    "mu", "n", "TU", "E_SOI", "M_SOI", "mu_earth", "masses", "T_min",
    "T_max_f", "dt_f", "T_max_b", "dt_b", "N_PLOT", "Integrator",
}
_WORKER_CONTEXT = None
_WORKER_STATES = None


def load_original_context(source_script):
    """Evaluate only original helpers, scalar constants, and event attributes."""
    source_script = Path(source_script)
    parsed = ast.parse(source_script.read_text())
    nodes = []
    functions = set()
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS:
            nodes.append(node)
            functions.add(node.name)
        elif isinstance(node, ast.Assign):
            scalar = all(isinstance(target, ast.Name) and target.id in _CONSTANTS
                         for target in node.targets)
            event = all(isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"earth_SOI_exit", "earth_perigee", "collision"}
                        and target.attr in {"terminal", "direction"}
                        for target in node.targets)
            if scalar or event:
                nodes.append(node)
    if functions != _FUNCTIONS:
        raise ValueError(f"Missing GA16 helpers: {sorted(_FUNCTIONS - functions)}")
    module = ast.Module(body=nodes, type_ignores=[])
    context = {"np": np, "Integration": Integration, "Transformations": Transformations}
    exec(compile(module, str(source_script), "exec"), context)
    if context["N_PLOT"] != 50:
        raise ValueError("This recovery expects the original run's N_PLOT=50.")
    digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode())
    digest.update(inspect.getsource(Integration.Integrator).encode())
    digest.update(inspect.getsource(Integration.CR3BP_ODE).encode())
    context["recovery_fingerprint"] = digest.hexdigest()
    return context


def _initialize_worker(source_script, states):
    global _WORKER_CONTEXT, _WORKER_STATES
    _WORKER_CONTEXT = load_original_context(source_script)
    _WORKER_STATES = states
    _WORKER_STATES.flags.writeable = False


def _integrate_index(index):
    result = _WORKER_CONTEXT["integrate_and_sample_one"](_WORKER_STATES[index])
    if result is None:
        raise RuntimeError(f"Previously retained trajectory at recovery index {index} was rejected.")
    features, curve, bounds = result
    return index, features, curve, np.asarray(bounds, dtype=np.float64)


def _array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    # Hash bounded contiguous slices, including non-contiguous source views.
    for start in range(0, len(array), 65536):
        block = np.ascontiguousarray(array[start:start + 65536])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


@dataclass
class OrbitCache:
    directory: Path
    source_ids: np.ndarray
    curves: np.ndarray
    bounds: np.ndarray
    features: np.ndarray
    done: np.ndarray
    context: dict

    def indices(self, source_ids):
        requested = np.asarray(source_ids, dtype=np.int64)
        indices = np.searchsorted(self.source_ids, requested)
        if np.any(indices >= len(self.source_ids)):
            raise ValueError("A requested orbit is absent from the recovery cache.")
        if not np.array_equal(self.source_ids[indices], requested):
            raise ValueError("A requested orbit is absent from the recovery cache.")
        if not np.all(self.done[indices]):
            raise ValueError("A requested orbit has not been integrated yet.")
        return indices

    def flush(self):
        for array in (self.curves, self.bounds, self.features, self.done):
            array.flush()


def _open_cache(directory, source_ids, states, context):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 1,
        "source_fingerprint": context["recovery_fingerprint"],
        "source_ids_sha256": _array_digest(source_ids),
        "states_sha256": _array_digest(states),
        "count": len(source_ids),
        "n_plot": context["N_PLOT"],
    }
    metadata_path = directory / "metadata.json"
    resume = metadata_path.exists()
    if resume and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Recovery cache belongs to different states or integration code; use a new cache directory.")
    specifications = {
        "curves": ((len(source_ids), 2, 2 * context["N_PLOT"] - 1), np.float64),
        "bounds": ((len(source_ids), 2), np.float64),
        "features": ((len(source_ids), 7), np.float64),
        "done": ((len(source_ids),), np.bool_),
    }
    arrays = {}
    for name, (shape, dtype) in specifications.items():
        path = directory / f"{name}.npy"
        if resume:
            arrays[name] = np.load(path, mmap_mode="r+", allow_pickle=False)
            if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
                raise ValueError(f"Invalid recovery cache array: {path}")
        else:
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if not resume:
        arrays["done"][:] = False
        arrays["done"].flush()
        np.save(directory / "source_ids.npy", source_ids, allow_pickle=False)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
    stored_ids = np.load(directory / "source_ids.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(stored_ids, source_ids):
        raise ValueError("Recovery source IDs differ from the cache.")
    return OrbitCache(directory, stored_ids, context=context, **arrays)


def _validate_result(index, curve, bounds, context, expected_times_days, expected_soi_positions_km):
    if not np.all(np.isfinite(curve)) or not np.all(np.isfinite(bounds)):
        raise RuntimeError(f"Non-finite recovered orbit at index {index}.")
    if expected_times_days is not None:
        actual_times = (bounds - context["T_min"]) * context["TU"] / context["day"]
        if not np.allclose(actual_times, expected_times_days[index], rtol=1e-9, atol=1e-8):
            raise RuntimeError(f"Recovered event times differ from the saved catalog at index {index}: "
                               f"{actual_times} versus {expected_times_days[index]}.")
    if expected_soi_positions_km is not None:
        x = curve[0, -1] + context["mu"]
        y = curve[1, -1]
        angle = bounds[1]
        position = context["d"] * np.array([
            np.cos(angle) * x - np.sin(angle) * y,
            np.sin(angle) * x + np.cos(angle) * y,
        ])
        if not np.allclose(position, expected_soi_positions_km[index], rtol=1e-9, atol=1e-4):
            raise RuntimeError(f"Recovered SOI position differs from the saved catalog at index {index}: "
                               f"{position} versus {expected_soi_positions_km[index]}.")


def build_orbit_cache(source_script, cache_dir, source_ids, states, *, workers=4,
                      batch_size=128, expected_times_days=None,
                      expected_soi_positions_km=None, on_batch=None,
                      priority_source_ids=None):
    """Integrate only unfinished rows and commit each bounded batch to disk.

    IDs must be sorted and unique.  Optional catalog arrays use the same order.
    ``on_batch(cache)`` runs after a completed batch, allowing ready pages to be
    rendered before the remaining integrations finish.
    """
    source_ids = np.asarray(source_ids, dtype=np.int64)
    states = np.asarray(states, dtype=np.float64)
    if states.shape != (len(source_ids), 6) or not len(source_ids):
        raise ValueError("Expected nonempty source IDs and matching (N, 6) states.")
    if np.any(np.diff(source_ids) <= 0) or not np.all(np.isfinite(states)):
        raise ValueError("Source IDs must be sorted/unique and states finite.")
    if workers < 1 or batch_size < 1:
        raise ValueError("Workers and batch size must be positive.")
    for expected in (expected_times_days, expected_soi_positions_km):
        if expected is not None and np.shape(expected) != (len(source_ids), 2):
            raise ValueError("Catalog validation arrays must have shape (N, 2).")
    source_script = str(Path(source_script).resolve())
    context = load_original_context(source_script)
    cache = _open_cache(cache_dir, source_ids, states, context)
    priority = np.empty(0, dtype=np.int64)
    if priority_source_ids is not None:
        requested = np.unique(np.asarray(priority_source_ids, dtype=np.int64))
        priority = np.searchsorted(source_ids, requested)
        if (np.any(priority >= len(source_ids))
                or not np.array_equal(source_ids[priority], requested)):
            raise ValueError("Priority source IDs are absent from the selected states.")
    completed = int(np.count_nonzero(cache.done))
    print(f"Orbit cache: {completed:,}/{len(source_ids):,} ready; workers={workers}", flush=True)
    if on_batch is not None:
        on_batch(cache)
    if completed == len(source_ids):
        return cache

    started = last_report = time.monotonic()
    initially_completed = completed

    def unfinished_batches():
        for start in range(0, len(priority), batch_size):
            indices = priority[start:start + batch_size]
            yield indices[~cache.done[indices]]
        for start in range(0, len(source_ids), batch_size):
            stop = min(start + batch_size, len(source_ids))
            yield np.flatnonzero(~cache.done[start:stop]) + start

    with get_context("fork").Pool(workers, initializer=_initialize_worker,
                                  initargs=(source_script, states)) as pool:
        for pending in unfinished_batches():
            if not len(pending):
                continue
            written = []
            for index, features, curve, bounds in pool.imap_unordered(_integrate_index, pending, chunksize=1):
                _validate_result(index, curve, bounds, context, expected_times_days,
                                 expected_soi_positions_km)
                cache.curves[index] = curve
                cache.bounds[index] = bounds
                cache.features[index] = features
                written.append(index)
                completed += 1
                now = time.monotonic()
                if now - last_report >= 30:
                    elapsed = now - started
                    rate = (completed - initially_completed) / elapsed
                    eta = (len(source_ids) - completed) / rate if rate else float("inf")
                    print(f"Integrated {completed:,}/{len(source_ids):,}; {rate:.1f}/s; "
                          f"elapsed {elapsed / 3600:.2f} h; estimated remaining {eta / 3600:.2f} h", flush=True)
                    last_report = now
            # Payload reaches disk before its completion markers become durable.
            for array in (cache.curves, cache.bounds, cache.features):
                array.flush()
            cache.done[written] = True
            cache.done.flush()
            if on_batch is not None:
                on_batch(cache)
    print(f"Orbit cache complete: {completed:,} trajectories", flush=True)
    return cache


def _save_new_figure(fig, path):
    from matplotlib import pyplot as plt
    from ga16_recover_feature_plots import valid_png
    path = Path(path)
    temporary = path.with_name(path.stem + ".recovering.png")
    try:
        if valid_png(path):
            return False
        fig.savefig(temporary, dpi=200, format="png")
        temporary.replace(path)
        print(f"Saved {path.name}", flush=True)
        return True
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def _inertial_curves(curves, bounds, context):
    """Vectorize the original position rotation with identical sampling times."""
    sample_count = context["N_PLOT"]
    times = np.concatenate((
        np.linspace(bounds[:, 0], context["T_min"], sample_count, axis=1),
        np.linspace(context["T_min"], bounds[:, 1], sample_count, axis=1)[:, 1:],
    ), axis=1)
    x = curves[:, 0] + context["mu"]
    y = curves[:, 1]
    cosine, sine = np.cos(times), np.sin(times)
    transformed = np.empty_like(curves)
    transformed[:, 0] = cosine * x - sine * y
    transformed[:, 1] = sine * x + cosine * y
    return transformed


def render_orbit_plots(cache, output_directory, selected_cluster_ids, cluster_ids,
                       cluster_sizes, representative_source_ids, *, ready_only=False,
                       source_script=None):
    """Create missing original-named orbit pages, retaining every supplied orbit.

    A page is rendered only when every selected orbit it needs is cached.
    Existing files are left untouched.  ``ready_only`` also permits incomplete
    representative curves; their figure is deferred until all are ready.
    """
    from ga16_recover_feature_plots import valid_png
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    selected_cluster_ids = np.asarray(selected_cluster_ids, dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)
    if selected_cluster_ids.shape != cache.source_ids.shape or cluster_sizes.shape != cluster_ids.shape:
        raise ValueError("Orbit labels and cluster sizes do not match their IDs.")
    if len(representative_source_ids) != len(cluster_ids):
        raise ValueError("Expected one representative source ID per cluster.")
    cmap = plt.get_cmap("turbo", max(len(cluster_ids), 1))
    colors = {int(cid): cmap(i) for i, cid in enumerate(cluster_ids)}
    context = cache.context
    if source_script is not None:
        reference = load_original_context(source_script)
        if reference["recovery_fingerprint"] != context["recovery_fingerprint"]:
            raise ValueError("Orbit rendering source differs from the integrated cache.")
    plotted = []
    # Group once, rather than scanning millions of labels for every page.
    order = np.argsort(selected_cluster_ids, kind="stable")
    ordered_labels = selected_cluster_ids[order]
    starts = np.searchsorted(ordered_labels, cluster_ids, side="left")
    stops = np.searchsorted(ordered_labels, cluster_ids, side="right")
    frames = ((False, "Rotating frame", ""),
              (True, f"Geocentric inertial frame (Moon at t = {context['T_min']:g} TU)",
               "Geocentric_inertial_"))

    for inertial, frame_name, prefix in frames:
        for first in range(0, len(cluster_ids), 24):
            last = min(first + 24, len(cluster_ids))
            path = output_directory / f"{prefix}clusters_{first // 24 + 1:03d}.png"
            if valid_png(path):
                continue
            groups = [order[starts[j]:stops[j]] for j in range(first, last)]
            if any(not np.all(cache.done[indices]) for indices in groups):
                if ready_only:
                    continue
                raise ValueError(f"Orbit page {path.name} has unfinished trajectories.")
            count = last - first
            columns = min(6, count)
            rows = int(np.ceil(count / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(18, 9),
                                     constrained_layout=True, squeeze=False)
            try:
                for ax, j, indices in zip(axes.ravel(), range(first, last), groups):
                    cid = int(cluster_ids[j])
                    # Each collection covers a bounded read; artists keep only
                    # this page's coordinate buffers, never the whole catalog.
                    for offset in range(0, len(indices), 4096):
                        block_ids = indices[offset:offset + 4096]
                        curves = np.asarray(cache.curves[block_ids])
                        if inertial:
                            curves = _inertial_curves(curves, cache.bounds[block_ids], context)
                        ax.add_collection(LineCollection(
                            curves.transpose(0, 2, 1), colors=[colors[cid]],
                            alpha=0.12, linewidths=0.7,
                        ))
                    context["plot_frame_bodies"](ax, inertial)
                    ax.autoscale_view()
                    total = int(cluster_sizes[j])
                    title = f"Cluster {cid} || {total} trajectories"
                    if len(indices) != total:
                        title += f"\nShown {len(indices)} / {total}"
                    ax.set_title(title)
                    ax.set(xlabel="X [LU]", ylabel="Y [LU]")
                    ax.set_aspect("equal")
                    ax.set_box_aspect(1)
                for ax in axes.ravel()[count:]:
                    ax.axis("off")
                fig.suptitle(f"{frame_name} - Cluster groups - Figure {first // 24 + 1}")
                if _save_new_figure(fig, path):
                    plotted.append(path)
            finally:
                plt.close(fig)
                ax = axes = fig = None
                gc.collect()

    representative_source_ids = np.asarray(representative_source_ids, dtype=np.int64)
    rep_indices = np.searchsorted(cache.source_ids, representative_source_ids)
    if (np.any(rep_indices >= len(cache.source_ids))
            or not np.array_equal(cache.source_ids[rep_indices], representative_source_ids)):
        raise ValueError("Representative source IDs are missing from the orbit cache.")
    if not np.all(cache.done[rep_indices]):
        if ready_only:
            return plotted
        raise ValueError("Representative orbits have not all been recovered.")
    for inertial, frame_name, suffix in frames:
        path = output_directory / f"representatives{suffix}.png"
        if valid_png(path):
            continue
        curves = np.asarray(cache.curves[rep_indices])
        if inertial:
            curves = _inertial_curves(curves, cache.bounds[rep_indices], context)
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        try:
            ax.add_collection(LineCollection(curves.transpose(0, 2, 1), colors="navy",
                                              alpha=0.2, linewidths=1, rasterized=True))
            context["plot_frame_bodies"](ax, inertial, size=50)
            ax.autoscale_view()
            ax.set(xlabel="X [LU]", ylabel="Y [LU]",
                   title=f"{frame_name} - {len(rep_indices)} medoids")
            ax.set_aspect("equal")
            ax.set_box_aspect(1)
            ax.grid(True, alpha=0.25)
            ax.legend()
            if _save_new_figure(fig, path):
                plotted.append(path)
        finally:
            plt.close(fig)
            ax = fig = None
            gc.collect()
    return plotted
