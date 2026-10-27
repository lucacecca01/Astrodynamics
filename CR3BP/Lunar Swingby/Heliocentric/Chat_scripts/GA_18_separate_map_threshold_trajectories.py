"""Apply absolute thresholds independently to maps 08, 10, 07, 06, and 05.

The saved >=1200-cell union contains every possible member of each separate
selection at >=1200. Reuse its exact per-map counts and 201-point CR3BP cache.
There is no union or intersection between maps when selecting the plots here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from PIL import Image

import GA_18_plots as plots
from GA_18_combined_trajectories import draw_combined, inertial_positions, trajectory_times
from GA_18_absolute_threshold_trajectories import DEFAULT_THRESHOLDS

MAPS = (
    ("08_hp_omega", 3, r"$h_p$–$\omega$", "quota–omega"),
    ("10_eps_theta", 5, r"$\varepsilon$–$\theta$", "energia–theta"),
    ("07_hp_e", 2, r"$h_p$–$e$", "quota–eccentricita"),
    ("06_eps_omega", 1, r"$\varepsilon$–$\omega$", "energia–omega"),
    ("05_eps_e", 0, r"$\varepsilon$–$e$", "energia–eccentricita"),
)


def select_one_map(counts, column, threshold):
    """Only the requested column participates in the selection."""
    return np.flatnonzero(counts[:, column] >= threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=plots.OUTPUT_DIR/"traiettorie_soglie_assolute_celle_1200_2400")
    parser.add_argument("--output", type=Path, default=plots.OUTPUT_DIR/"traiettorie_soglie_separate_per_mappa")
    parser.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    parent = json.loads((args.source/"riepilogo.json").read_text())
    parent_minimum = min(map(int, parent["thresholds"]))
    if min(thresholds) < parent_minimum:
        parser.error(f"La selezione sorgente permette soglie >= {parent_minimum}.")
    with np.load(args.source/f"selezione_ge_{parent_minimum}.npz", allow_pickle=False) as data:
        members, counts = data["members"], data["per_map_counts"]
        ids = data["source_ids"]
    cache = Path(parent["dense_cache"])
    synodic = np.load(cache/"curves.npy", mmap_mode="r", allow_pickle=False)
    cached_ids = np.load(cache/"source_ids.npy", mmap_mode="r", allow_pickle=False)
    done = np.load(cache/"done.npy", mmap_mode="r", allow_pickle=False)
    if (not np.array_equal(cached_ids, ids) or not done.all()
            or synodic.shape != (len(members), 2, 201) or counts.shape != (len(members), 6)):
        raise ValueError("Cache o conteggi non coerenti con i membri della selezione sorgente.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("source_id duplicati.")
    # Common spatial limits across all maps, thresholds and full-size frames.
    low = synodic.min(axis=(0, 2))
    upper = synodic.max(axis=(0, 2))
    for start in range(0, len(members), 4096):
        stop = min(start+4096, len(members))
        times = trajectory_times(members[start:stop, 5:7], 101)
        frame = inertial_positions(synodic[start:stop], times)
        low = np.minimum(low, frame.min(axis=(0, 2)))
        upper = np.maximum(upper, frame.max(axis=(0, 2)))
    limits = np.array([low, upper])
    center, side = limits.mean(axis=0), np.max(limits[1]-limits[0])*1.06
    cmap = plt.get_cmap("viridis")
    d = plots.EARTH_MOON_DISTANCE_KM
    args.output.mkdir(parents=True, exist_ok=True)
    # Saved global maxima also cover cells absent from the >=1200 union.
    original_maps = json.loads((plots.OUTPUT_DIR/"traiettorie_top_10_percent_201_punti/riepilogo.json").read_text())["maps"]
    summary = dict(source=str(args.source.resolve()), samples_per_trajectory=201,
                   selection="Soglia applicata solo alla cella della singola mappa, nessuna aggregazione fra mappe",
                   thresholds=thresholds, same_axes_opacity=True, color_scale_scope="per_map",
                   same_color_scale_across_thresholds=True, maps={})
    for key, column, label, plain_label in MAPS:
        output = args.output/key
        output.mkdir(parents=True, exist_ok=True)
        map_scores = counts[:, column]
        max_occurrences = int(original_maps[key[:2]]["maximum_occurrences"])
        normalization = Normalize(vmin=min(thresholds), vmax=max(min(thresholds)+1, max_occurrences))
        overview, axes = plt.subplots(2, 4, figsize=(20, 11), layout="constrained")
        overview_axes = axes.ravel()
        map_summary = dict(label=plain_label, maximum_occurrences=max_occurrences,
                           color_min=float(normalization.vmin), color_max=float(normalization.vmax), thresholds={})
        for threshold, ax in zip(thresholds, overview_axes):
            indices = select_one_map(counts, column, threshold)
            row_members, row_scores = members[indices], map_scores[indices]
            curves = synodic[indices]
            count = len(indices)
            title = (f"Solo mappa {key[:2]}: {label} — celle con almeno {threshold} asteroidi\n"
                     f"{count:,} traiettorie distinte, 201 punti ciascuna; massimo della mappa: {max_occurrences}".replace(",", "."))
            draw_combined(curves, row_members, row_scores, f"Asteroidi nella cella: {plain_label}",
                          output/f"traiettorie_ge_{threshold}_tre_riferimenti.png", parent["population_count"],
                          title=title, normalization=normalization, fixed_alpha=0.12, global_limits=limits)
            order = np.argsort(row_scores, kind="stable")
            for start in range(0, count, 2048):
                chosen = order[start:start+2048]
                colors = cmap(normalization(row_scores[chosen]))
                colors[:, 3] = 0.12
                ax.add_collection(LineCollection(curves[chosen].transpose(0, 2, 1), colors=colors,
                                                 linewidths=0.25, rasterized=True))
            ax.set_aspect("equal")
            ax.set_xlim(center[0]-side/2, center[0]+side/2)
            ax.set_ylim(center[1]-side/2, center[1]+side/2)
            ax.set_xlabel("x sinodica geocentrica [km]")
            ax.set_ylabel("y [km]")
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
            ax.grid(alpha=0.15)
            ax.add_patch(Circle((0, 0), plots.R_EARTH_KM, color="#2563eb", zorder=5))
            ax.add_patch(Circle((d, 0), 1737, color="#64748b", zorder=5))
            ax.annotate("Terra", (0, 0), xytext=(4, 5), textcoords="offset points", fontsize=8)
            ax.annotate("Luna", (d, 0), xytext=(4, 5), textcoords="offset points", fontsize=8)
            ax.set_title(f"≥ {threshold} asteroidi per cella\n{count:,} traiettorie".replace(",", "."))
            if count == 0:
                ax.text(0.5, 0.7, f"Nessuna cella ammessa\nMassimo: {max_occurrences}",
                        transform=ax.transAxes, ha="center", bbox=dict(facecolor="white", edgecolor="none"))
            np.savez_compressed(output/f"selezione_ge_{threshold}.npz", source_ids=ids[indices],
                                cluster_ids=row_members[:, 1].astype(int), occurrences=row_scores)
            map_summary["thresholds"][str(threshold)] = count
            print(f"{key}: soglia {threshold}, {count} traiettorie", flush=True)
        for ax in overview_axes[len(thresholds):]:
            ax.axis("off")
        overview.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=cmap),
                          ax=list(overview_axes[:len(thresholds)]), shrink=0.8,
                          label=f"Asteroidi nella cella: {plain_label}")
        overview.suptitle(f"Solo mappa {key[:2]}: {label} — selezioni indipendenti dalle altre mappe", fontsize=17)
        overview.supxlabel("201 punti per curva; stessa scala spaziale e trasparenza. Scala colori specifica per coppia, fissa fra le sue soglie.\n"
                           "Raggio al perigeo espresso come quota sopra la superficie terrestre.", fontsize=10)
        overview.savefig(output/"00_confronto_sinodico_soglie.png", dpi=180)
        plt.close(overview)
        (output/"riepilogo.json").write_text(json.dumps(map_summary, indent=2)+"\n")
        summary["maps"][key] = map_summary
        pages = []
        for threshold in thresholds:
            with Image.open(output/f"traiettorie_ge_{threshold}_tre_riferimenti.png") as image:
                pages.append(image.convert("RGB"))
        try:
            pages[0].save(output/"tutte_le_soglie_tre_riferimenti.pdf", "PDF", save_all=True,
                          append_images=pages[1:], resolution=180, quality=95, subsampling=0)
        finally:
            for image in pages:
                image.close()
    (args.output/"riepilogo.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Completate cinque serie separate in {args.output}", flush=True)


if __name__ == "__main__":
    main()
