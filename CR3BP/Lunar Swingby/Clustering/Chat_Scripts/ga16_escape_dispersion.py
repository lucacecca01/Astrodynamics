"""SOI speed/direction dispersion for GA16 plotting and plot recovery.

The catalog contains geocentric inertial velocities at the finite Earth SOI.
These functions do not extrapolate those velocities to asymptotic v-infinity.
"""

import ast
from contextlib import nullcontext
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


METRIC_NAMES = (
    "STD |v SOI| [km/s]",
    "SOI direction dispersion [0, 1]",
)
RANKING_TITLE = "SOI escape velocity dispersion by cluster"


def escape_dispersion(velocities):
    """Return population STD of speed and dispersion of unit directions.

    Direction dispersion is ``1 - norm(mean(v / norm(v)))``. It is zero
    for aligned directions and one for a zero resultant. For planar data it
    is the circular variance, so the 0/360-degree boundary is continuous.
    Zero-speed members enter the speed STD but have no defined direction.
    An empty input, or no nonzero velocity, has NaN direction dispersion.
    """
    velocities = np.asarray(velocities, dtype=float)
    if (velocities.ndim != 2 or velocities.shape[1] not in (2, 3)
            or not np.all(np.isfinite(velocities))):
        raise ValueError("Expected finite SOI velocities with shape (N, 2) or (N, 3).")
    if not len(velocities):
        return np.array([np.nan, np.nan])
    speeds = np.linalg.norm(velocities, axis=1)
    if not np.isfinite(speeds).all():
        raise ValueError("Non-finite SOI speed modulus.")
    moving = speeds > 0
    direction = np.nan
    if np.any(moving):
        unit = velocities[moving] / speeds[moving, None]
        direction = np.clip(1 - np.linalg.norm(unit.mean(axis=0)), 0., 1.)
    return np.array([np.std(speeds, ddof=0), direction])


def recover_escape_rankings(catalog, cluster_ids, groups, source_script, output_directory):
    """Regenerate the original-named rankings with SOI dispersion metrics.

    Read the plotting helpers with AST, without importing or running GA16's
    integration/clustering main. Stage complete PDF/PNG/CSV files before
    replacing the previous orbital-element rankings in the output directory.
    Return the paths of all published files.
    """
    catalog = np.asarray(catalog)
    cluster_ids = np.asarray(cluster_ids)
    if (catalog.ndim != 2 or catalog.shape[1] != 12
            or cluster_ids.ndim != 1 or len(cluster_ids) != len(groups)
            or not np.issubdtype(cluster_ids.dtype, np.integer)
            or len(np.unique(cluster_ids)) != len(cluster_ids)):
        raise ValueError("Expected a 12-column catalog and aligned unique integer cluster IDs/groups.")
    if not len(cluster_ids):
        return []
    sizes = np.empty(len(groups), dtype=np.int64)
    values = np.empty((len(groups), 2), dtype=float)
    for j, (cid, group) in enumerate(zip(cluster_ids, groups)):
        group = np.asarray(group)
        if (group.ndim != 1 or not len(group) or not np.issubdtype(group.dtype, np.integer)
                or np.any(group < 0) or np.any(group >= len(catalog))
                or not np.all(catalog[group, 1] == cid)):
            raise ValueError(f"Invalid member rows for cluster {cid}.")
        sizes[j] = len(group)
        values[j] = escape_dispersion(catalog[group, 2:5])

    source_script = Path(source_script)
    helpers = {"save_figure", "plot_dispersion_rankings"}
    nodes = [node for node in ast.parse(source_script.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in helpers]
    if {node.name for node in nodes} != helpers:
        raise ValueError("GA16 source is missing the dispersion-ranking plot helpers.")
    module = ast.Module(body=nodes, type_ignores=[])
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"orbital_dispersion_ranking_K{len(cluster_ids)}"
    published = []
    with TemporaryDirectory(prefix=".escape_ranking_", dir=directory) as temporary:
        stage = Path(temporary)
        scope = dict(np=np, plt=plt, PdfPages=PdfPages, nullcontext=nullcontext,
                     SAVE=True, PLOT_CLUSTERS=False, output_directory=stage)
        exec(compile(module, str(source_script), "exec"), scope)
        scope["plot_dispersion_rankings"](
            cluster_ids, sizes, values, METRIC_NAMES, RANKING_TITLE, stem,
        )
        with (stage / f"{stem}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("cluster_id", "n_trajectories", "std_speed_soi_km_s",
                             "direction_dispersion_1_minus_mean_unit_norm"))
            writer.writerows((int(cid), int(size), *metrics)
                             for cid, size, metrics in zip(cluster_ids, sizes, values))
        for path in sorted(stage.iterdir()):
            target = directory / path.name
            os.replace(path, target)
            published.append(target)
    return published
