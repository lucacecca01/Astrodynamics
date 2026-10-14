"""Recover GA16 feature figures from the saved 12-column exit catalog.

This module does not import the production sweep or reintegrate trajectories.
Every assigned catalog row is plotted; completed PNG files are preserved.
"""

from multiprocessing import get_context
from pathlib import Path
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import ndtri
from threadpoolctl import threadpool_limits


FEATURE_NAMES = (r"$\varepsilon$ [km$^2$/s$^2$]", "e [-]", r"$\omega$ [deg]")
FEATURE_PAIRS = ((0, 1), (0, 2), (1, 2))
_WORKER_CONTEXT = None
_THREAD_LIMITER = None


def valid_png(path):
    """Return whether a completed PNG exists, including its chunk integrity."""
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def _save_and_close(fig, path):
    """Publish only a complete figure and always release its artists."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
        fig.savefig(temporary, dpi=200, format="png")
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def gaussian_shape_penalty(parameters):
    """Use the production standardized Q-Q error and circular omega center."""
    data = np.array(parameters, dtype=float, copy=True)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("Servono almeno due righe finite [eps, e, omega_deg].")
    angular_mean = np.mean(np.exp(1j * np.deg2rad(data[:, 2])))
    origin = np.rad2deg(np.angle(angular_mean))
    data[:, 2] = (data[:, 2] - origin + 180) % 360 - 180
    std = np.std(data, axis=0)
    variable = (np.ptp(data, axis=0) > 0) & (std > 0)
    standardized = np.divide(
        data - data.mean(axis=0), std, out=np.zeros_like(data), where=variable,
    )
    normal = ndtri((np.arange(len(data)) + 0.5) / len(data))
    penalty = np.mean((np.sort(standardized, axis=0) - normal[:, None])**2, axis=0)
    penalty[~variable] = 1.0
    if abs(angular_mean) < 1e-8:
        penalty[2] = 1.0
    return penalty


def _orbital_features(catalog, members):
    values = catalog[members, 9:12].copy()
    values[:, 2] %= 360
    return values


def recover_feature_overviews(catalog, group_ids, member_groups,
                              representative_ids, output_directory):
    """Recover the 2-D and 3-D overviews separately to limit peak memory."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = len(group_ids)
    cmap = plt.get_cmap("turbo", max(count, 1))
    style = dict(
        fontsize=8, color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )
    created = []
    for dimension in (2, 3):
        path = directory / f"features_{dimension}D_K{count}.png"
        if valid_png(path):
            continue
        print(f"Recovering {path.name} with all assigned points", flush=True)
        if dimension == 2:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        else:
            fig = plt.figure(figsize=(10, 8), constrained_layout=True)
            ax3 = fig.add_subplot(111, projection="3d")
        try:
            for color_id, (cid, members, representative_id) in enumerate(
                zip(group_ids, member_groups, representative_ids)
            ):
                values = _orbital_features(catalog, members)
                representative = catalog[representative_id, 9:12].copy()
                representative[2] %= 360
                color = cmap(color_id)
                if dimension == 2:
                    for ax, (i, j) in zip(axes, FEATURE_PAIRS):
                        ax.scatter(values[:, i], values[:, j], s=3, alpha=0.3,
                                   color=color, rasterized=True)
                        ax.annotate(str(cid), representative[[i, j]], xytext=(4, 4),
                                    textcoords="offset points", **style)
                else:
                    ax3.scatter(values[:, 0], values[:, 1], values[:, 2],
                                s=3, alpha=0.3, color=color, depthshade=False)
                    ax3.text(*representative, str(cid), **style)
            if dimension == 2:
                for ax, (i, j) in zip(axes, FEATURE_PAIRS):
                    ax.set_xlabel(FEATURE_NAMES[i])
                    ax.set_ylabel(FEATURE_NAMES[j])
                    ax.grid(alpha=0.3)
                fig.suptitle(f"Feature projections — K={count}")
            else:
                ax3.set_xlabel(FEATURE_NAMES[0])
                ax3.set_ylabel(FEATURE_NAMES[1])
                ax3.set_zlabel(FEATURE_NAMES[2])
                ax3.set_title(f"Feature space — K={count}")
            _save_and_close(fig, path)
            created.append(path)
        finally:
            plt.close(fig)
    return created


def recover_cluster_feature_plots(catalog, cid, members, color, output_directory):
    """Recover one cluster's orbital and SOI velocity/time figures."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    path = directory / f"cluster_{cid:03d}_features.png"
    if not valid_png(path):
        values = _orbital_features(catalog, members)
        penalty = gaussian_shape_penalty(values) if len(values) >= 2 else np.ones(3)
        names = list(FEATURE_NAMES)
        angular_mean = np.mean(np.exp(1j * np.deg2rad(values[:, 2])))
        angle_defined = abs(angular_mean) > 1e-8
        if angle_defined:
            origin = np.rad2deg(np.angle(angular_mean))
            values[:, 2] = (values[:, 2] - origin + 180) % 360 - 180
            names[2] = r"$\Delta\omega$ [deg]"
        stds = np.std(values, axis=0)
        if not angle_defined:
            stds[2] = np.nan
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        try:
            for ax, (i, j) in zip(axes[0], FEATURE_PAIRS):
                ax.scatter(values[:, i], values[:, j], s=5, alpha=0.35,
                           color=color, rasterized=True)
                ax.set_xlabel(names[i])
                ax.set_ylabel(names[j])
                ax.grid(alpha=0.3)
            for j, ax in enumerate(axes[1]):
                ax.hist(values[:, j], bins=30, color=color, alpha=0.8)
                ax.set_xlabel(names[j])
                ax.set_ylabel("Number of trajectories")
                ax.set_title(f"STD = {stds[j]:.5g} | Gaussian Q-Q = {penalty[j]:.4g}")
                ax.grid(alpha=0.3)
            fig.suptitle(f"Cluster {cid} — {len(values)} trajectories")
            _save_and_close(fig, path)
            created.append(path)
        finally:
            plt.close(fig)

    path = directory / f"cluster_{cid:03d}_velocity_time.png"
    if not valid_png(path):
        velocities = catalog[members, 2:4]
        times = catalog[members, 6]
        mean_velocity = velocities.mean(axis=0)
        centered_velocity = velocities - mean_velocity
        _, _, axes_velocity = np.linalg.svd(centered_velocity, full_matrices=False)
        pc1_axis = axes_velocity[0]
        if pc1_axis[np.argmax(np.abs(pc1_axis))] < 0:
            pc1_axis = -pc1_axis
        pc1 = centered_velocity @ pc1_axis
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        try:
            axes[0].scatter(velocities[:, 0], velocities[:, 1], s=10, alpha=0.65,
                            color=color, rasterized=True)
            axes[0].scatter(*mean_velocity, marker="x", s=60, color="black",
                            label="Media del cluster", zorder=3)
            axes[0].set(xlabel=r"$v_x$ [km/s]", ylabel=r"$v_y$ [km/s]",
                        title="Velocità alla SOI — geocentriche inerziali")
            axes[0].set_aspect("equal", adjustable="datalim")
            axes[1].scatter(pc1, times, s=10, alpha=0.65, color=color, rasterized=True)
            axes[1].scatter(0, times.mean(), marker="x", s=60, color="black",
                            label="Media del cluster", zorder=3)
            axes[1].set(xlabel="PC1 delle velocità centrate [km/s]",
                        ylabel="Tempo ETD → SOI [giorni]",
                        title="Tempo di uscita — PCA del singolo cluster")
            for ax in axes:
                ax.set_box_aspect(1)
                ax.grid(alpha=0.3)
                ax.legend(fontsize="small")
            fig.suptitle(f"Cluster {cid} — {len(velocities)} traiettorie")
            _save_and_close(fig, path)
            created.append(path)
        finally:
            plt.close(fig)
    return created


def _initialize_worker(catalog, group_ids, member_groups, output_directory):
    global _WORKER_CONTEXT, _THREAD_LIMITER
    catalog = catalog.view()
    catalog.flags.writeable = False
    _WORKER_CONTEXT = (catalog, group_ids, member_groups, output_directory)
    _THREAD_LIMITER = threadpool_limits(limits=1)


def _render_group(index):
    catalog, group_ids, member_groups, directory = _WORKER_CONTEXT
    color = plt.get_cmap("turbo", max(len(group_ids), 1))(index)
    return recover_cluster_feature_plots(
        catalog, int(group_ids[index]), member_groups[index], color, directory,
    )


def recover_feature_plots(catalog, group_ids, member_groups, representative_ids,
                          output_directory, *, workers=4):
    """Recover missing figures, sharing the read-only catalog across fork workers.

    ``member_groups`` and ``representative_ids`` are aligned with sorted
    ``group_ids``. The catalog columns follow ``trajectory_exit_catalog.txt``.
    Return paths of newly generated figures; existing valid PNGs are untouched.
    """
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if catalog.ndim != 2 or catalog.shape[1] != 12:
        raise ValueError("Expected the saved 12-column trajectory exit catalog")
    if not (len(group_ids) == len(member_groups) == len(representative_ids)):
        raise ValueError("Group IDs, members, and representatives must be aligned")
    with threadpool_limits(limits=1):
        created = recover_feature_overviews(
            catalog, group_ids, member_groups, representative_ids, output_directory,
        )
        pending = [i for i, cid in enumerate(group_ids) if any(
            not valid_png(Path(output_directory) / f"cluster_{cid:03d}_{suffix}.png")
            for suffix in ("features", "velocity_time")
        )]
        if workers == 1:
            colors = plt.get_cmap("turbo", max(len(group_ids), 1))
            for completed, index in enumerate(pending, start=1):
                created.extend(recover_cluster_feature_plots(
                    catalog, int(group_ids[index]), member_groups[index],
                    colors(index), output_directory,
                ))
                print(f"Feature clusters recovered: {completed}/{len(pending)}", flush=True)
        elif pending:
            with get_context("fork").Pool(
                processes=min(workers, len(pending)), initializer=_initialize_worker,
                initargs=(catalog, group_ids, member_groups, output_directory),
            ) as pool:
                for completed, paths in enumerate(
                    pool.imap_unordered(_render_group, pending, chunksize=1), start=1,
                ):
                    created.extend(paths)
                    print(f"Feature clusters recovered: {completed}/{len(pending)}", flush=True)
    return created
