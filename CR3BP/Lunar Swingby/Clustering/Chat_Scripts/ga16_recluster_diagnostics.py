"""Recover GA16 diagnostics from real saved reclustering stages.

This module does not run integrations or clustering. The catalog retains the
12-column GA16 exit format; each supplied label array describes an actual
saved stage. Curves follow that chronological order, not a fictitious K scan.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import textwrap

from matplotlib import pyplot as plt
import numpy as np

from ga16_escape_dispersion import escape_dispersion
from ga16_recover_feature_plots import (
    _save_and_close, gaussian_shape_penalty, valid_png,
)


_VERSION = "eps-e-saved-stage-diagnostics-v1"


def _digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    for start in range(0, len(array), 65536):
        digest.update(np.ascontiguousarray(array[start:start+65536]).tobytes())
    return digest.hexdigest()


def _catalog_digest(catalog):
    digest = hashlib.sha256(str(catalog.shape).encode())
    for start in range(0, len(catalog), 65536):
        block = np.asarray(catalog[start:start+65536])[:, (0, 2, 3, 4, 9, 10, 11)]
        digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def _normalize(values):
    """Match GA16's scalar global normalization, including unassigned rows."""
    centered = values - np.mean(values, axis=0)
    scale = np.linalg.norm(centered) / np.sqrt(len(centered))
    return np.zeros_like(centered) if scale <= 1e-14 else centered / scale


def _stage_statistics(catalog, features, name, labels, sigma_limit):
    ordered = np.flatnonzero(labels >= 0)
    ordered = ordered[np.argsort(labels[ordered], kind="stable")]
    if not len(ordered):
        raise ValueError(f"Stage {name!r} has no assigned trajectories.")
    groups = np.split(ordered, np.flatnonzero(np.diff(labels[ordered])) + 1)
    records, weighted_qq, physical = [], [], []
    weighted_variance = 0.
    outliers = np.zeros(3, dtype=int)
    for ids in groups:
        values = features[ids]
        variance = np.var(values, axis=0)
        squared = np.sum((values-values.mean(axis=0))**2, axis=1)
        sigma_squared = float(variance.sum())
        bad = [bool(np.any(squared > factor**2 * sigma_squared))
               for factor in (2., 3., sigma_limit)]
        outliers += bad
        # Own copy: circular centering must never alter the saved catalog.
        quantities = np.column_stack((catalog[ids, 9:12], catalog[ids, 2:4]))
        qq = gaussian_shape_penalty(quantities[:, :3]) if len(ids) >= 2 else np.ones(3)
        angle_mean = np.mean(np.exp(1j*np.deg2rad(quantities[:, 2])))
        if abs(angle_mean) < 1e-8:
            quantities[:, 2] = np.nan
        else:
            origin = np.rad2deg(np.angle(angle_mean))
            quantities[:, 2] = (quantities[:, 2]-origin+180) % 360-180
        std = np.std(quantities, axis=0, ddof=0)
        escape = escape_dispersion(catalog[ids, 2:5])
        records.append(dict(
            cluster_id=int(labels[ids[0]]), n_trajectories=len(ids),
            std_eps_km2_s2=float(std[0]), std_e=float(std[1]), std_omega_deg=float(std[2]),
            std_vx_soi_km_s=float(std[3]), std_vy_soi_km_s=float(std[4]),
            std_speed_soi_km_s=float(escape[0]), direction_dispersion=float(escape[1]),
            gaussian_qq_eps=float(qq[0]), gaussian_qq_e=float(qq[1]), gaussian_qq_omega=float(qq[2]),
            outlier_above_2sigma=bad[0], outlier_above_3sigma=bad[1],
            outlier_above_max_sigma=bad[2],
        ))
        physical.append(std)
        weighted_qq.append(qq)
        weighted_variance += len(ids)*float(variance.mean())
    sizes = np.array([record["n_trajectories"] for record in records])
    physical = np.asarray(physical)
    count = np.sum(np.isfinite(physical), axis=0)
    physical_mean = np.divide(np.nansum(physical, axis=0), count,
                              out=np.full(5, np.nan), where=count > 0)
    small = [int(np.count_nonzero(mask)) for mask in (sizes == 1, sizes < 10, sizes < 100)]
    return dict(
        name=str(name), n_clusters=len(groups), assigned=int(sizes.sum()),
        unassigned=int(np.count_nonzero(labels < 0)), minimum_size=int(sizes.min()),
        normalized_within_cluster_std=float(np.sqrt(weighted_variance/sizes.sum())),
        mean_physical_std=physical_mean.tolist(),
        weighted_gaussian_qq=np.average(weighted_qq, axis=0, weights=sizes).tolist(),
        outlier_cluster_counts=outliers.tolist(),
        outlier_cluster_percent=(100*outliers/len(groups)).tolist(),
        small_cluster_counts=small,
        small_cluster_percent=(100*np.asarray(small)/len(groups)).tolist(),
        clusters=records,
    )


def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _publish_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def recover_diagnostics(catalog, stages, output_directory, *, sigma_limit=2.5):
    """Generate the four diagnostic PNGs, plus auditable stage/cluster tables.

    ``stages`` is a chronological list of ``(name, labels)`` pairs. Features
    are epsilon/eccentricity normalized globally over every catalog row.
    Physical STD is averaged equally across clusters; Q-Q and normalized
    within-cluster variance are weighted by cluster size, as in production.
    Return the newly generated PNG paths; identical complete figures are kept.
    """
    catalog = np.asarray(catalog)
    stages = list(stages)
    if (catalog.ndim != 2 or catalog.shape[1] != 12 or not len(catalog)
            or not np.isfinite(catalog[:, 9:11]).all() or not stages
            or np.isnan(sigma_limit) or sigma_limit <= 0):
        raise ValueError("Expected a finite eps/e exit catalog, saved stages, and positive sigma limit.")
    parsed = []
    for name, labels in stages:
        labels = np.asarray(labels)
        if (labels.shape != (len(catalog),) or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < -1)):
            raise ValueError(f"Invalid labels for stage {name!r}.")
        parsed.append((str(name), labels))
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    provenance = dict(version=_VERSION, catalog_sha256=_catalog_digest(catalog),
                      stages=[dict(name=name, labels_sha256=_digest(labels)) for name, labels in parsed],
                      sigma_limit=float(sigma_limit))
    json_path = directory / "diagnostic_saved_stages.json"
    matching = False
    if json_path.exists():
        try:
            matching = json.loads(json_path.read_text()).get("provenance") == provenance
        except (ValueError, OSError):
            pass
    final_k = len(np.unique(parsed[-1][1][parsed[-1][1] >= 0]))
    names = ("gaussian_shape_vs_stage.png", "physical_STD_and_3sigma_vs_K.png",
             f"cluster_diagnostics_K{final_k}.png", "STD_vs_K.png")
    tables = (directory / "diagnostic_stages.csv", directory / "diagnostic_clusters.csv")
    if matching and all(valid_png(directory/name) for name in names) and all(p.exists() for p in tables):
        return []
    features = np.column_stack((_normalize(catalog[:, 9:10]), _normalize(catalog[:, 10:11])))
    statistics = []
    for name, labels in parsed:
        result = _stage_statistics(catalog, features, name, labels, sigma_limit)
        statistics.append(result)
        print(f"Diagnostics: {name}: K={result['n_clusters']}, assigned={result['assigned']:,}", flush=True)
    x = np.arange(1, len(statistics)+1)
    ticks = [f"{i}\nK={s['n_clusters']}" for i, s in zip(x, statistics)]
    description = textwrap.fill("; ".join(f"{i}: {s['name']}" for i, s in zip(x, statistics)), width=145)
    created = []

    def stage_axis(ax):
        ax.set_xticks(x, ticks, fontsize=8)
        ax.set_xlabel("Stadio salvato (ordine effettivo)")
        ax.grid(alpha=.3)

    def finish(fig, filename):
        fig.text(.5, .018, description, ha="center", va="bottom", fontsize=8)
        path = directory / filename
        if not matching or not valid_png(path):
            _save_and_close(fig, path)
            created.append(path)
        else:
            plt.close(fig)

    qq = np.array([s["weighted_gaussian_qq"] for s in statistics])
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=.23, top=.89)
    for j, name in enumerate(("eps", "e", "omega")):
        ax.plot(x, qq[:, j], "o-", label=name)
    ax.set(ylabel="Weighted mean Gaussian Q-Q error (lower is better)", ylim=(0, None))
    stage_axis(ax)
    ax.legend()
    ax.set_title("Gaussian shape — actual saved stages")
    finish(fig, names[0])

    history = np.array([s["mean_physical_std"] for s in statistics])
    outliers = np.array([s["outlier_cluster_counts"] for s in statistics])
    small = np.array([s["small_cluster_counts"] for s in statistics])
    k = np.array([s["n_clusters"] for s in statistics])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(bottom=.20, top=.90, hspace=.48, wspace=.30)
    axes = axes.ravel()
    for j, name in enumerate(("eps [km^2/s^2]", "e [-]", "omega [deg]")):
        axes[j].plot(x, history[:, j], "o-")
        axes[j].set_ylabel(f"Mean STD — {name}")
    axes[3].plot(x, history[:, 3], "o-", label="vx")
    axes[3].plot(x, history[:, 4], "o-", label="vy")
    axes[3].set_ylabel("Mean STD — v SOI [km/s]")
    for target, factor in ((axes[4], np.ones(len(k))), (axes[5], 100/k)):
        for values, color, label in ((outliers[:, 0], "darkorange", "> 2 sigma"),
                                      (outliers[:, 1], "darkred", "> 3 sigma"),
                                      (small[:, 1], "blue", "< 10 trajectories"),
                                      (small[:, 2], "green", "< 100 trajectories")):
            target.plot(x, values*factor, "o-", color=color, label=label)
        target.set_title("Radial outliers and cluster sizes")
    axes[4].set_ylabel("Number of clusters")
    axes[5].set_ylabel("Clusters [%]")
    axes[5].set_ylim(0, 100)
    for ax in axes:
        stage_axis(ax)
        ax.set_ylim(bottom=0)
    for ax in axes[3:]:
        ax.legend(fontsize=8)
    fig.suptitle("Physical dispersion — actual saved stages, not an ordered K scan")
    finish(fig, names[1])

    final = statistics[-1]
    records = final["clusters"]
    fields = ("std_eps_km2_s2", "std_e", "std_omega_deg", "std_speed_soi_km_s",
              "direction_dispersion", "n_trajectories")
    labels = ("STD eps [km^2/s^2]", "STD e [-]", "STD omega [deg]",
              "STD |v SOI| [km/s]", "SOI direction dispersion [0, 1]", "Number of trajectories")
    ids = np.array([r["cluster_id"] for r in records])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(bottom=.19, top=.90, hspace=.40, wspace=.30)
    for ax, field, label in zip(axes.ravel(), fields, labels):
        ax.scatter(ids, [record[field] for record in records], s=15)
        ax.set(xlabel="Cluster ID", ylabel=label, ylim=(0, None))
        ax.grid(alpha=.3)
    axes.ravel()[4].set_ylim(0, 1)
    fig.suptitle(f"Initial compactness and SOI escape diversity — K={final_k} — {final['name']}")
    finish(fig, names[2])

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(bottom=.23, top=.89)
    ax.plot(x, [s["normalized_within_cluster_std"] for s in statistics], "o-", color="blue")
    stage_axis(ax)
    ax.set(ylabel="Within-cluster STD — normalized eps/e", ylim=(0, None),
           title="Cluster dispersion — actual saved stages")
    finish(fig, names[3])

    stage_fields = ("stage", "name", "n_clusters", "assigned", "unassigned", "minimum_size",
                    "normalized_within_cluster_std", "mean_std_eps", "mean_std_e", "mean_std_omega",
                    "mean_std_vx_soi", "mean_std_vy_soi", "weighted_qq_eps", "weighted_qq_e", "weighted_qq_omega",
                    "clusters_above_2sigma", "clusters_above_3sigma", "clusters_above_max_sigma",
                    "clusters_size_1", "clusters_size_below_10", "clusters_size_below_100")
    temporary = tables[0].with_name(tables[0].name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(stage_fields)
        for i, s in enumerate(statistics, 1):
            writer.writerow((i, s["name"], s["n_clusters"], s["assigned"], s["unassigned"], s["minimum_size"],
                             s["normalized_within_cluster_std"], *s["mean_physical_std"],
                             *s["weighted_gaussian_qq"], *s["outlier_cluster_counts"], *s["small_cluster_counts"]))
    os.replace(temporary, tables[0])
    temporary = tables[1].with_name(tables[1].name + ".tmp")
    with temporary.open("w", newline="") as stream:
        fields = ("stage", "stage_name", *records[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(statistics, 1):
            writer.writerows(dict(stage=i, stage_name=s["name"], **record) for record in s["clusters"])
    os.replace(temporary, tables[1])
    report = dict(provenance=provenance, n_catalog_rows=len(catalog),
                  normalization="eps/e each normalized on all catalog rows, including unassigned members",
                  physical_std_average="unweighted across clusters; omega circularly centered, undefined angles omitted",
                  gaussian_qq_average="weighted by cluster member count",
                  within_cluster_std="sqrt(size-weighted mean of per-cluster mean normalized eps/e variance)",
                  figure_order="Chronological supplied stages; K is annotated, never sorted",
                  stages=statistics, plots=list(names))
    _publish_text(json_path, json.dumps(_clean_json(report), indent=2, allow_nan=False)+"\n")
    return created
