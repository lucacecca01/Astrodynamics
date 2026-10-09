"""Rotate GA_16 Earth-SOI exits and propagate them in the planar Sun-Earth CR3BP.

Run without arguments to use the most recently written GA_16 export. Each input
trajectory is launched from every 10-degree point on the Earth SOI, preserving
its geocentric inertial speed and its signed angle relative to the local tangent.
Propagation and plots use heliocentric coordinates with nonrotating axes.
The initial heliocentric velocity includes Earth's orbital velocity vector.
Time and coordinates use Sun-Earth CR3BP units; 4*pi TU is two Earth periods,
not necessarily two revolutions of each spacecraft. Collisions stop propagation.

Input columns: cluster_id source_id v_soi_km_s exit_angle_deg position_angle_deg.
An older four-column export requires an explicit --reference-position-deg;
that option assigns a common reference position instead of the original exits.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import binary_closing, binary_dilation


# Match GA_16's physical constants and Laplace sphere of influence.
G = 6.67430e-20                  # km^3 kg^-1 s^-2
M_SUN = 1.989e30                 # kg
M_EARTH = 5.974e24               # kg
LU = 1.496e8                    # km
MU = M_EARTH / (M_SUN + M_EARTH)
TU = np.sqrt(LU**3 / (G * (M_SUN + M_EARTH)))  # s
VU = LU / TU                    # km/s
SOI_RADIUS = (M_EARTH / M_SUN)**(2 / 5)        # LU
R_SUN = 695700.0 / LU
R_EARTH = 6378.0 / LU
FINAL_TIME = 4 * np.pi           # Two Earth revolutions in Sun-Earth units.

INPUT_FILE = None                # None selects the newest GA_16 TXT.
ANGLE_STEP_DEG = 10.0
N_SAMPLES = 1000
WORKERS = max(1, min(8, os.cpu_count() or 1))
RTOL = 1e-10
ATOL = 1e-12
MAX_STEP = 0.05                  # TU; independent of output sampling.
COVERAGE_RESOLUTION = 700
SAVE_TRAJECTORIES = False        # Full runs can produce many GB of coordinates.
ALGORITHM_VERSION = "ga17-heliocentric-4pi-v2"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _disk(radius):
    """Return a compact disk-shaped structuring element in grid cells."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def trajectory_coverage(paths, *, resolution=600, reference_points=()):
    """Rasterize all finite segments into a square grid in the input length units.

    ``paths`` has shape ``(n_paths, n_samples, 2)``.  NaNs separate finite path
    runs, so no artificial segment connects points across an integration gap.
    Consecutive stored points are linearly interpolated at most half a cell
    apart before local closing/dilation.  A sparsely sampled curved trajectory
    therefore remains limited by the temporal sampling of the supplied paths.

    ``reference_points`` are included in the plot bounds, but not the mask.
    The mask uses ``origin='lower'`` and extent ``(xmin, xmax, ymin, ymax)``.
    """
    # Keep the caller's float32 storage: large clusters can contain more than
    # 100,000 paths, so promoting the entire array would consume extra gigabytes.
    paths = np.asarray(paths)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("paths must have shape (n_paths, n_samples, 2)")
    if isinstance(resolution, (bool, np.bool_)) or int(resolution) != resolution:
        raise ValueError("coverage resolution must be an integer")
    resolution = int(resolution)
    if resolution < 32:
        raise ValueError("coverage resolution must be at least 32")

    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    bounds_batch_size = max(1, 262144 // max(paths.shape[1], 1))
    for start in range(0, len(paths), bounds_batch_size):
        block = paths[start : start + bounds_batch_size]
        finite_points = block[np.isfinite(block).all(axis=-1)]
        if finite_points.size:
            lower = np.minimum(lower, finite_points.min(axis=0))
            upper = np.maximum(upper, finite_points.max(axis=0))
        del block, finite_points
    if not np.isfinite(lower).all():
        raise ValueError("Cannot plot trajectories without any finite points")
    references = np.asarray(reference_points, dtype=float)
    if references.size:
        references = references.reshape(-1, 2)
        if not np.isfinite(references).all():
            raise ValueError("reference_points must contain only finite values")
        lower = np.minimum(lower, references.min(axis=0))
        upper = np.maximum(upper, references.max(axis=0))

    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1e-8)
    # Equal physical cell width in both directions and room for local dilation.
    span *= 1.10
    lower = center - 0.5 * span
    upper = center + 0.5 * span
    cell_size = span / resolution
    extent = np.array([lower[0], upper[0], lower[1], upper[1]])
    mask = np.zeros((resolution, resolution), dtype=bool)

    # Bounded batches prevent an unusually long segment or large cluster from
    # producing one temporary array for the entire interpolated data set.
    for path in paths:
        finite = np.isfinite(path).all(axis=-1)
        # Include isolated finite samples without materializing one global copy
        # of all finite points or all point-to-cell indices.
        point_cells = np.floor((path[finite] - lower) / cell_size).astype(np.int64)
        point_cells = np.clip(point_cells, 0, resolution - 1)
        mask[point_cells[:, 1], point_cells[:, 0]] = True
        valid_segments = finite[:-1] & finite[1:]
        indices = np.flatnonzero(valid_segments)
        for start in range(0, indices.size, 4096):
            batch = indices[start : start + 4096]
            a = (path[batch] - lower) / cell_size
            delta = (path[batch + 1] - path[batch]) / cell_size
            steps = np.maximum(1, np.ceil(2.0 * np.max(np.abs(delta), axis=1)))
            steps = steps.astype(np.int64)
            counts = steps + 1
            segment_ids = np.repeat(np.arange(batch.size), counts)
            starts = np.cumsum(counts) - counts
            sample_ids = np.arange(int(counts.sum())) - np.repeat(starts, counts)
            weights = sample_ids / steps[segment_ids]
            cells = np.floor(a[segment_ids] + delta[segment_ids] * weights[:, None])
            cells = np.clip(cells.astype(np.int64), 0, resolution - 1)
            mask[cells[:, 1], cells[:, 0]] = True

    # Preserve the original occupancy even near a domain edge, where closing
    # alone could remove cells.  No filling of arbitrary enclosed holes occurs.
    closed = binary_closing(mask, structure=_disk(2))
    mask = binary_dilation(mask | closed, structure=_disk(1))
    return {
        "coverage_mask": mask,
        "coverage_extent": extent,
        "coverage_cell_size": cell_size,
        "coverage_method": (
            "Union of finite, linearly interpolated trajectory segments; "
            "half-cell raster sampling; disk closing radius 2 cells; "
            "disk dilation radius 1 cell; no convex hull or global hole filling"
        ),
    }


def plot_cluster_trajectories(
    paths,
    cluster_id,
    n_original,
    launch_count,
    soi_radius,
    output_path,
    *,
    coverage_resolution=600,
    full_horizon=None,
    collision_count=0,
    show=False,
):
    """Save heliocentric inertial trajectories and their sampled spatial footprint.

    Coordinates use the Sun--Earth distance as the length unit (1 AU).  Every
    supplied path is drawn, including trajectories ending at a collision.
    The Sun is at the origin; Earth and its SOI are marked at t=0 only.
    ``output_path`` selects the output format through its extension, e.g. PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    paths = np.asarray(paths)
    earth_x = 1.0
    coverage = trajectory_coverage(
        paths,
        resolution=coverage_resolution,
        reference_points=(
            (-1.0, -1.0),
            (1.0, 1.0),
            (earth_x - soi_radius, -soi_radius),
            (earth_x + soi_radius, soi_radius),
        ),
    )
    extent = coverage["coverage_extent"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    color = "#2673ab"
    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        shade = np.ma.masked_where(~coverage["coverage_mask"], coverage["coverage_mask"])
        cmap = ListedColormap([color])
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        ax.imshow(
            shade,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=0.20,
            zorder=1,
        )
        ax.add_collection(
            LineCollection(
                paths,
                colors=color,
                linewidths=0.30,
                alpha=0.18,
                zorder=2,
                rasterized=True,
            )
        )
        ax.scatter([0.0], [0.0], s=115, color="#f5bd25", edgecolor="#a4770a", zorder=5)
        ax.scatter([earth_x], [0.0], s=24, color="#173b73", zorder=5)
        ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="gray",
                            linewidth=0.8, linestyle=":", zorder=3))
        ax.add_patch(
            Circle(
                (earth_x, 0.0),
                soi_radius,
                fill=False,
                color="#173b73",
                linewidth=0.8,
                linestyle="--",
                zorder=4,
            )
        )
        ax.set(
            xlim=extent[:2],
            ylim=extent[2:],
            xlabel="Heliocentric inertial x [AU]",
            ylabel="Heliocentric inertial y [AU]",
            title=(
                f"Cluster {cluster_id}: {n_original} exit velocities × "
                f"{launch_count} SOI positions\n"
                f"{paths.shape[0]} propagated trajectories; duration: {FINAL_TIME / (2*np.pi):g} Earth periods"
                + (f"\nFull duration: {full_horizon}; collisions: {collision_count}"
                   if full_horizon is not None else "")
            ),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.16, linewidth=0.5)
        ax.legend(
            handles=[
                Line2D([], [], color=color, linewidth=0.8, label="Propagated trajectories"),
                Patch(facecolor=color, alpha=0.20, label="Approximate sampled footprint"),
                Line2D([], [], marker="o", color="none", markerfacecolor="#f5bd25", label="Sun"),
                Line2D([], [], marker="o", color="none", markerfacecolor="#173b73", label="Earth at t=0"),
                Line2D([], [], color="gray", linestyle=":", linewidth=0.8, label="Earth orbit"),
                Line2D([], [], color="#173b73", linestyle="--", linewidth=0.8, label="Earth SOI at t=0"),
            ],
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )
        fig.text(
            0.5,
            0.015,
            f"Shade: local grid approximation; cell size "
            f"{coverage['coverage_cell_size']:.3g} AU "
            f"({coverage_resolution} × {coverage_resolution} cells).",
            ha="center",
            fontsize=8,
            color="#555555",
        )
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        fig.savefig(output_path, dpi=180)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return coverage


def find_input_file():
    if INPUT_FILE is not None:
        return Path(INPUT_FILE).expanduser().resolve()
    candidates = list((SCRIPT_DIRECTORY / "Clusters").glob(
        "GA_16_plots_K*/cluster_exit_velocities.txt"))
    if not candidates:
        raise FileNotFoundError("Nessun TXT GA_16 trovato. Eseguire GA_16 o specificare --input.")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def load_exit_data(path, reference_position_deg=None):
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0 or data.shape[1] not in (4, 5):
        raise ValueError("Il TXT deve contenere righe con 4 o 5 colonne del formato GA_16.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Il TXT contiene velocita', angoli o identificativi non finiti.")
    if np.any(data[:, :2] < 0) or np.any(data[:, :2] != np.floor(data[:, :2])):
        raise ValueError("cluster_id e source_id devono essere interi non negativi.")
    if len(np.unique(data[:, 1])) != len(data):
        raise ValueError("source_id duplicati: ogni traiettoria originale deve apparire una sola volta.")
    if np.any(data[:, 2] <= 0):
        raise ValueError("I moduli delle velocita' di uscita devono essere positivi.")
    if reference_position_deg is not None:
        if not np.isfinite(reference_position_deg):
            raise ValueError("L'angolo della posizione di riferimento deve essere finito.")
        data = np.column_stack((data[:, :4], np.full(len(data), reference_position_deg)))
    elif data.shape[1] == 4:
        raise ValueError(
            "Il TXT contiene solo 4 colonne: manca position_angle_deg, necessario per "
            "conservare l'angolo originale rispetto alla tangente. Rigenerare il TXT "
            "con GA_16 aggiornato. Solo per assumere una posizione comune, usare "
            "esplicitamente --reference-position-deg 0 (punto +x).")
    data[:, 3:5] %= 360.0
    return data[np.argsort(data[:, 0], kind="stable")]


def launch_angles(step_deg):
    if not np.isfinite(step_deg) or not 0 < step_deg <= 360:
        raise ValueError("Il passo angolare deve essere compreso tra 0 e 360 gradi.")
    count = int(round(360 / step_deg))
    if not np.isclose(count * step_deg, 360, atol=1e-9, rtol=0):
        raise ValueError("Il passo angolare deve dividere esattamente 360 gradi.")
    return np.arange(count) * step_deg


def make_initial_states(rows, angles_deg):
    """Heliocentric states at t=0; add Earth's velocity to every rotated SOI exit."""
    theta = np.deg2rad(np.asarray(angles_deg))[None, :]
    alpha_minus_phi = np.deg2rad(rows[:, 3] - rows[:, 4])[:, None]
    beta = theta + alpha_minus_phi
    speed = rows[:, 2, None] / VU
    rx, ry = SOI_RADIUS * np.cos(theta), SOI_RADIUS * np.sin(theta)
    states = np.empty((len(rows), len(angles_deg), 4))
    states[:, :, 0] = 1 + rx
    states[:, :, 1] = ry
    # Earth is at (1, 0), with heliocentric velocity (0, 1) in normalized units.
    # Vector sum: v_helio = v_Earth + v_geocentric; there is no frame rotation here.
    states[:, :, 2] = speed * np.cos(beta)
    states[:, :, 3] = 1.0 + speed * np.sin(beta)
    return states.reshape(-1, 4)


def cr3bp_rhs(t, state):
    """Sun-Earth restricted dynamics in heliocentric, nonrotating coordinates."""
    x, y, vx, vy = state
    earth_x, earth_y = np.cos(t), np.sin(t)
    dx_earth, dy_earth = x - earth_x, y - earth_y
    r_sun3 = (x*x + y*y)**1.5
    r_earth3 = (dx_earth*dx_earth + dy_earth*dy_earth)**1.5
    # Subtract the Sun's acceleration (the indirect term) to keep the origin
    # heliocentric while retaining the same circular restricted three-body model.
    return np.array((vx, vy,
                     -(1 - MU) * x / r_sun3 - MU * (dx_earth / r_earth3 + earth_x),
                     -(1 - MU) * y / r_sun3 - MU * (dy_earth / r_earth3 + earth_y)))


def sun_collision(t, state):
    return np.hypot(state[0], state[1]) - R_SUN


def earth_collision(t, state):
    return np.hypot(state[0] - np.cos(t), state[1] - np.sin(t)) - R_EARTH


sun_collision.terminal = earth_collision.terminal = True
sun_collision.direction = earth_collision.direction = -1


def jacobi_constant(states, times=0.0):
    """Evaluate the CR3BP invariant after converting heliocentric states to rotating axes."""
    states = np.asarray(states)
    x, y, vx, vy = np.moveaxis(states, -1, 0)
    cosine, sine = np.cos(times), np.sin(times)
    ux, uy = cosine*x + sine*y, -sine*x + cosine*y
    wx, wy = cosine*vx + sine*vy + uy, -sine*vx + cosine*vy - ux
    return ((ux - MU)**2 + uy*uy + 2 * (1 - MU) / np.hypot(ux, uy)
            + 2 * MU / np.hypot(ux - 1, uy) - wx*wx - wy*wy)


def propagate_one(task):
    initial_state, samples, rtol, atol, max_step = task
    solution = solve_ivp(cr3bp_rhs, (0, FINAL_TIME), initial_state, method="DOP853",
                         rtol=rtol, atol=atol, max_step=max_step, dense_output=True,
                         events=(sun_collision, earth_collision))
    if not solution.success:
        raise RuntimeError(f"Integrazione fallita per {initial_state.tolist()}: {solution.message}")
    stop_time = float(solution.t[-1])
    status = 0                       # 0: full horizon; 1: Sun; 2: Earth.
    for body, event_times in enumerate(solution.t_events, start=1):
        if len(event_times):
            status = body
    times = np.linspace(0, stop_time, samples)
    states = solution.sol(times).T
    if not np.all(np.isfinite(states)):
        raise RuntimeError("Stati non finiti nella propagazione Sole-Terra.")
    paths = states[:, :2].astype(np.float32)
    jacobi_error = float(np.max(np.abs(jacobi_constant(states, times) - jacobi_constant(initial_state))))
    return paths, stop_time, status, len(times), jacobi_error


def cluster_fingerprint(rows, settings):
    digest = hashlib.sha256(ALGORITHM_VERSION.encode())
    digest.update(np.ascontiguousarray(rows, dtype=np.float64).tobytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def completed_cluster(metadata_path, fingerprint, plot_path, data_path):
    if not metadata_path.exists() or not plot_path.exists() or not data_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        return (isinstance(metadata, dict) and metadata.get("fingerprint") == fingerprint
                and metadata.get("complete") is True)
    except (ValueError, OSError):
        return False


def run_cluster(rows, angles, output_directory, settings, pool=None, *, force=False):
    cluster_id = int(rows[0, 0])
    stem = output_directory / f"cluster_{cluster_id:03d}"
    plot_path, data_path, metadata_path = (stem.with_suffix(suffix) for suffix in (".png", ".npz", ".json"))
    fingerprint = cluster_fingerprint(rows, settings)
    if not force and completed_cluster(metadata_path, fingerprint, plot_path, data_path):
        print(f"Cluster {cluster_id}: gia' completato, riuso i risultati.", flush=True)
        return
    metadata_path.unlink(missing_ok=True)

    start = time.perf_counter()
    initial_states = make_initial_states(rows, angles)
    count = len(initial_states)
    paths = np.empty((count, settings["samples"], 2), dtype=np.float32)
    end_times = np.empty(count)
    statuses = np.empty(count, dtype=np.int8)
    sample_counts = np.empty(count, dtype=np.int32)
    jacobi_errors = np.empty(count)
    print(f"Cluster {cluster_id}: {len(rows)} x {len(angles)} = {count} propagazioni", flush=True)
    tasks = ((state, settings["samples"], settings["rtol"], settings["atol"], settings["max_step"])
             for state in initial_states)
    results = map(propagate_one, tasks) if pool is None else pool.imap(propagate_one, tasks, chunksize=8)
    last_report = start
    for i, result in enumerate(results):
        paths[i], end_times[i], statuses[i], sample_counts[i], jacobi_errors[i] = result
        now = time.perf_counter()
        if now - last_report >= 15 or i + 1 == count:
            print(f"  {i + 1}/{count}; {now - start:.1f} s", flush=True)
            last_report = now

    coverage = plot_cluster_trajectories(
        paths, cluster_id, len(rows), len(angles), SOI_RADIUS, plot_path,
        coverage_resolution=settings["coverage_resolution"],
        full_horizon=int(np.count_nonzero(statuses == 0)),
        collision_count=int(np.count_nonzero(statuses != 0)),
    )
    saved = dict(
        input_rows=rows, initial_states=initial_states,
        source_ids=np.repeat(rows[:, 1].astype(np.int64), len(angles)),
        launch_angles_deg=np.tile(angles, len(rows)),
        end_times=end_times, statuses=statuses, sample_counts=sample_counts,
        sample_fractions=np.linspace(0, 1, settings["samples"]),
        jacobi_errors=jacobi_errors, **coverage,
    )
    if settings["save_trajectories"]:
        saved["trajectories_xy"] = paths
    # Write completion metadata last so an interrupted cluster is recomputed.
    np.savez_compressed(data_path, **saved)
    metadata = dict(
        complete=True, fingerprint=fingerprint, cluster_id=cluster_id,
        input_trajectories=len(rows), propagated_trajectories=count,
        full_horizon=int(np.count_nonzero(statuses == 0)),
        sun_collisions=int(np.count_nonzero(statuses == 1)),
        earth_collisions=int(np.count_nonzero(statuses == 2)),
        max_jacobi_error=float(jacobi_errors.max()),
        elapsed_seconds=time.perf_counter() - start, settings=settings,
        frame="Heliocentric inertial axes, Sun at origin; x/y in LU, time in TU",
        LU_km=LU, TU_seconds=TU, VU_km_s=VU,
        status_codes={"0": "Final time reached (default 4*pi, two Earth periods)",
                      "1": "Sun collision", "2": "Earth collision"},
        sampling="Per-trajectory time = sample_fractions * end_times[i]; includes the exact final/collision point.",
        coverage_method=coverage["coverage_method"],
    )
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    print(f"  Salvato {plot_path.name}; durata completa: {metadata['full_horizon']}; "
          f"collisioni Sole/Terra: {metadata['sun_collisions']}/{metadata['earth_collisions']}; "
          f"max |delta C| = {metadata['max_jacobi_error']:.3g}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="TXT esportato da GA_16")
    parser.add_argument("--output", type=Path, help="Cartella dei risultati GA_17")
    parser.add_argument("--clusters", type=int, nargs="+", help="Solo gli ID indicati; default: tutti")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--angle-step", type=float, default=ANGLE_STEP_DEG)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--coverage-resolution", type=int, default=COVERAGE_RESOLUTION)
    parser.add_argument("--reference-position-deg", type=float,
                        help="Assumi questa posizione iniziale comune anziche' le posizioni originali")
    parser.add_argument("--save-trajectories", action="store_true", default=SAVE_TRAJECTORIES,
                        help="Salva anche tutte le coordinate campionate nei file NPZ")
    parser.add_argument("--force", action="store_true", help="Ricalcola anche i cluster gia' completati")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.samples < 2 or args.coverage_resolution < 32:
        parser.error("Servono workers >= 1, samples >= 2 e coverage-resolution >= 32.")
    try:
        path = args.input.expanduser().resolve() if args.input else find_input_file()
        data = load_exit_data(path, args.reference_position_deg)
        angles = launch_angles(args.angle_step)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    all_clusters = np.unique(data[:, 0].astype(int))
    if args.clusters is not None:
        missing = set(args.clusters) - set(all_clusters)
        if missing:
            parser.error(f"Cluster assenti dal TXT: {sorted(missing)}")
        data = data[np.isin(data[:, 0], args.clusters)]
    clusters = np.unique(data[:, 0].astype(int))
    output = (args.output or SCRIPT_DIRECTORY / "Clusters" / f"GA_17_heliocentric_from_{path.parent.name}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(angle_step_deg=args.angle_step, samples=args.samples, rtol=RTOL, atol=ATOL,
                    max_step=MAX_STEP, final_time=FINAL_TIME, mu=MU, soi_radius=SOI_RADIUS,
                    lu_km=LU, tu_seconds=TU, sun_radius=R_SUN, earth_radius=R_EARTH,
                    coverage_resolution=args.coverage_resolution,
                    save_trajectories=args.save_trajectories,
                    reference_position_deg=args.reference_position_deg)
    print(f"Input: {path}\nOutput: {output}\n"
          f"{len(clusters)} cluster, {len(data)} traiettorie originali, "
          f"{len(data) * len(angles)} propagazioni eliocentriche inerziali; "
          f"t_finale = {FINAL_TIME / np.pi:g}pi TU Sole-Terra "
          f"({FINAL_TIME / (2*np.pi):g} periodi terrestri, {FINAL_TIME * TU / 86400:.3f} giorni).", flush=True)
    if args.reference_position_deg is not None:
        print(f"Posizione originale comune ASSUNTA: {args.reference_position_deg:g} gradi.", flush=True)
    if args.workers == 1:
        for cluster_id in clusters:
            run_cluster(data[data[:, 0] == cluster_id], angles, output, settings, force=args.force)
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for cluster_id in clusters:
                run_cluster(data[data[:, 0] == cluster_id], angles, output, settings, pool, force=args.force)
    print(f"Completato. Grafici e dati in {output}", flush=True)


if __name__ == "__main__":
    main()
