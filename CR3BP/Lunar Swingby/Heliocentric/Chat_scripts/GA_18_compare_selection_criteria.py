"""Compare upper 1% by winning-cluster counts with upper-10%-cell union.

Reuse previously reconstructed 201-point curves. Both rows have the same
coordinate limits, cluster colors, line width, and opacity.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

import GA_18_plots as plots
from GA_18_combined_trajectories import trajectory_times, inertial_positions


BASE = plots.OUTPUT_DIR


def read_selection(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "riepilogo.json").read_text())
    with np.load(directory / "membri_selezionati.npz", allow_pickle=False) as data:
        members = data["members"]
        scores = data["scores"] if "scores" in data else None
    cache = Path(metadata["dense_cache"])
    curves = np.load(cache / "curves.npy", mmap_mode="r", allow_pickle=False)
    cached_ids = np.load(cache / "source_ids.npy", mmap_mode="r", allow_pickle=False)
    done = np.load(cache / "done.npy", mmap_mode="r", allow_pickle=False)
    if (not np.array_equal(cached_ids, members[:, 0].astype(np.int64))
            or not done.all() or curves.shape != (len(members), 2, 201)):
        raise ValueError("La selezione non corrisponde alla cache completa a 201 punti.")
    return metadata, members, scores, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=BASE / "traiettorie_combinate_top_10_percent_clusters_201_punti")
    parser.add_argument("--cells", type=Path, default=BASE / "traiettorie_top_10_percent_201_punti")
    parser.add_argument("--output", type=Path, default=BASE / "confronto_upper_1_percent_cluster_upper_10_percent_celle")
    args = parser.parse_args()
    cluster_meta, all_cluster_members, all_cluster_scores, cluster_curves = read_selection(args.clusters)
    cell_meta, cell_members, _, cell_curves = read_selection(args.cells)
    if (cluster_meta["method"] != "clusters" or cell_meta["top_percent_of_each_map_scale"] != 10
            or cluster_meta["catalog"] != cell_meta["catalog"]
            or cluster_meta["report"] != cell_meta["report"]):
        raise ValueError("Le selezioni non provengono dagli stessi dati o criteri attesi.")
    count = int(np.ceil(0.01 * cluster_meta["population_count"]))
    if len(all_cluster_members) < count:
        raise ValueError("Il risultato precedente non contiene l'upper 1% completo.")
    chosen = np.lexsort((all_cluster_members[:, 0], -all_cluster_scores))[:count]
    chosen = chosen[np.argsort(all_cluster_members[chosen, 0])]
    cluster_members = all_cluster_members[chosen]
    cluster_scores = all_cluster_scores[chosen]
    curve_a = cluster_curves[chosen]
    curve_b = np.asarray(cell_curves)
    report = Path(cluster_meta["report"]).read_text()
    frequency = np.bincount([int(x) for x in re.findall(r"Miglior trasferimento: cluster (\d+)", report)])
    if not np.array_equal(cluster_scores, frequency[cluster_members[:, 1].astype(int)]):
        raise ValueError("I punteggi non corrispondono ai cluster vincenti del report corrente.")
    cell_scores = frequency[cell_members[:, 1].astype(int)]
    overlap, ia, ib = np.intersect1d(cluster_members[:, 0], cell_members[:, 0], return_indices=True)
    if len(overlap) and (not np.array_equal(cluster_members[ia], cell_members[ib])
                         or not np.allclose(curve_a[ia], curve_b[ib], rtol=0, atol=1e-5)):
        raise ValueError("Traiettorie condivise incoerenti tra i due risultati.")
    print(f"Upper 1% cluster: {count}, soglia {cluster_scores.min()} asteroidi; "
          f"unione celle upper 10%: {len(cell_members)}; comuni: {len(overlap)}.", flush=True)
    rows = [(cluster_members, curve_a, cluster_scores), (cell_members, curve_b, cell_scores)]
    union_clusters = np.unique(np.concatenate((cluster_members[:, 1], cell_members[:, 1])))
    cmap = plt.get_cmap("turbo")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), layout="constrained")
    limits = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
    for row, (members, curves, scores) in enumerate(rows):
        order = np.argsort(scores, kind="stable")
        for start in range(0, len(members), 2048):
            indices = order[start:start+2048]
            times = trajectory_times(members[indices, 5:7], 101)
            chunk = curves[indices]
            frames = (chunk, inertial_positions(chunk, times), inertial_positions(chunk, times, lunar_origin=True))
            colors = cmap(np.searchsorted(union_clusters, members[indices, 1])/max(1, len(union_clusters)-1))
            colors[:, 3] = 0.12
            for col, (ax, frame) in enumerate(zip(axes[row], frames)):
                ax.add_collection(LineCollection(frame.transpose(0, 2, 1), colors=colors,
                                                 linewidths=0.35, rasterized=True))
                if col < 2:
                    points = frame.transpose(0, 2, 1).reshape(-1, 2)
                    limits[0] = np.minimum(limits[0], points.min(axis=0))
                    limits[1] = np.maximum(limits[1], points.max(axis=0))
    center = limits.mean(axis=0)
    side = np.max(limits[1]-limits[0])*1.06
    d = plots.EARTH_MOON_DISTANCE_KM
    moon_soi = (plots.MOON_MASS_KG/5.974e24)**0.4*d
    titles = ("Sinodico Terra–Luna", "Geocentrico inerziale", "Lunocentrico, assi inerziali — zoom")
    descriptions = (f"1% per asteroidi del cluster vincente — {count:,} traiettorie".replace(",", "."),
                    f"Unione celle upper 10% delle mappe 05–10 — {len(cell_members):,} traiettorie".replace(",", "."))
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.set_title(titles[col], fontsize=13)
            ax.set_xlabel("x [km]")
            ax.set_ylabel("y [km]")
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
            if col < 2:
                ax.set_xlim(center[0]-side/2, center[0]+side/2)
                ax.set_ylim(center[1]-side/2, center[1]+side/2)
                ax.add_patch(Circle((0, 0), plots.R_EARTH_KM, color="#2563eb", zorder=5))
                ax.annotate("Terra", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=8)
                if col == 0:
                    ax.add_patch(Circle((d, 0), 1737, color="#64748b", zorder=5))
                    ax.annotate("Luna", (d, 0), xytext=(5, 5), textcoords="offset points", fontsize=8)
                    ax.add_patch(Circle((d, 0), moon_soi, fill=False, ls="--", color="#888888", lw=0.7))
                else:
                    ax.add_patch(Circle((0, 0), d, fill=False, ls="--", color="#888888", lw=0.7))
            else:
                ax.set_xlim(-100000, 100000)
                ax.set_ylim(-100000, 100000)
                ax.add_patch(Circle((0, 0), 1737, color="#64748b", zorder=5))
                ax.annotate("Luna", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=8)
                ax.add_patch(Circle((0, 0), moon_soi, fill=False, ls="--", color="#888888", lw=0.7))
        axes[row, 1].text(0.5, 1.15, descriptions[row], transform=axes[row, 1].transAxes,
                         ha="center", fontsize=15, weight="bold")
    fig.suptitle("Confronto dei due criteri di selezione", fontsize=19)
    fig.supxlabel("201 punti per traiettoria. Colori per cluster, identici nelle due righe; stessi assi, spessore e trasparenza.\n"
                  "Dal perigeo alla SOI terrestre; assi inerziali allineati al sinodico a ETD, origine lunare aggiornata a ogni epoca.", fontsize=11)
    args.output.mkdir(parents=True, exist_ok=True)
    figure_path = args.output / "confronto_due_criteri_tre_riferimenti.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    for filename, (members, _, scores) in zip(("upper_1_percent_cluster", "upper_10_percent_celle"), rows):
        np.savez_compressed(args.output/f"{filename}.npz", members=members, scores=scores,
                            source_ids=members[:, 0].astype(np.int64))
    metadata = dict(cluster_percentage=1, cluster_population=cluster_meta["population_count"],
                    cluster_selected=count, cluster_cutoff=int(cluster_scores.min()),
                    tie_break="source_id crescente", cell_percentage_of_each_map_scale=10,
                    cell_selected=len(cell_members), common_sources=len(overlap),
                    samples_per_trajectory=201, cluster_parent=str(args.clusters.resolve()),
                    cell_parent=str(args.cells.resolve()), same_rendering_parameters=True,
                    colors="cluster", axes_identical=True)
    (args.output/"riepilogo.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Salvata figura: {figure_path}", flush=True)


if __name__ == "__main__":
    main()
