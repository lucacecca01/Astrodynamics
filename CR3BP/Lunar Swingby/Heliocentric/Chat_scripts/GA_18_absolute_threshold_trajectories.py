"""Union of maps 05--10 with absolute asteroid-count thresholds on cells."""

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

import GA_18_plots as plots
import GA_18_high_occurrence_trajectories as high
from GA_18_combined_trajectories import combined_scores, draw_combined, inertial_positions, trajectory_times

DEFAULT_THRESHOLDS = (1200, 1400, 1600, 1800, 2000, 2200, 2400)


def threshold_selections(per_map_counts, thresholds):
    """An OR across maps, never a sum across maps; every source appears once."""
    maximum = per_map_counts.max(axis=1)
    return maximum, {int(t): np.flatnonzero(maximum >= t) for t in thresholds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--report", type=Path, default=plots.OUTPUT_DIR/"riepilogo.txt")
    parser.add_argument("--cache", type=Path, default=high.DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2)-2)))
    parser.add_argument("--output", type=Path, default=plots.OUTPUT_DIR/"traiettorie_soglie_assolute_celle_1200_2400")
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or min(thresholds) < 1 or args.workers < 1:
        parser.error("Soglie e numero di processi devono essere positivi.")
    report = args.report.read_text()
    stats_path = Path(re.search(r"^Catalogo: (.+)$", report, re.M)[1])
    member_path = stats_path.with_name("trajectory_exit_catalog.txt")
    if not re.search(r"Occorrenze migliori trasferimenti: asteroids;", report):
        raise ValueError("Le soglie in asteroidi richiedono il conteggio asteroids, non trajectories.")
    stats = plots.load_table(stats_path, 14)
    stats = stats[np.argsort(stats[:, 0])]
    print("Calcolo conteggi delle sei celle per ciascun membro vincente...", flush=True)
    members = plots.load_table(member_path, 12)
    winner_ids = np.array([int(x) for x in re.findall(r"Miglior trasferimento: cluster (\d+)", report)], dtype=int)
    winner_rows = np.searchsorted(stats[:, 0], winner_ids)
    if not np.array_equal(stats[winner_rows, 0], winner_ids):
        raise ValueError("Cluster vincenti assenti dalle statistiche.")
    only_bound = "Orbita di parcheggio: legata, perigeo sopra la superficie" in report
    candidates, counts = combined_scores(members, stats, winner_rows, member_path, only_bound, plots.N_BINS)
    del members
    maximum, selections = threshold_selections(counts, thresholds)
    for threshold, indices in selections.items():
        print(f"≥ {threshold} asteroidi per cella: {len(indices)} traiettorie uniche; "
              f"per mappa {np.count_nonzero(counts >= threshold, axis=0).tolist()}", flush=True)
    selected = selections[thresholds[0]]
    population_count = len(candidates)
    members = candidates[selected]
    per_map_counts = counts[selected]
    maximum = maximum[selected]
    del candidates, counts
    if len(members):
        sparse, sparse_inertial, _, old_n_plot = high.checked_curves(args.cache, members, allow_missing=True)
        del sparse_inertial
        n_plot = max(101, 2*old_n_plot)
        synodic, inertial, directions, dense_cache = high.dense_curves(args.cache, members, sparse, n_plot, args.workers)
        del sparse
        low = np.minimum(synodic.min(axis=(0, 2)), inertial.min(axis=(0, 2)))
        upper = np.maximum(synodic.max(axis=(0, 2)), inertial.max(axis=(0, 2)))
        del inertial
    else:
        n_plot = 101
        synodic = np.empty((0, 2, 201))
        directions, dense_cache = np.empty(0), None
        low, upper = np.array([-1000000, -1000000]), np.array([1000000, 1000000])
    global_limits = np.array([low, upper])
    normalization = Normalize(vmin=thresholds[0], vmax=max(thresholds[0]+1, int(maximum.max(initial=0))))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    columns = min(4, len(thresholds))
    rows = int(np.ceil(len(thresholds)/columns))
    overview, axes = plt.subplots(rows, columns, figsize=(5*columns, 5*rows+1),
                                  layout="constrained", squeeze=False)
    center = global_limits.mean(axis=0)
    side = np.max(global_limits[1]-global_limits[0])*1.06
    summary = dict(definition="Membro in almeno una cella con >= N asteroidi, unione delle mappe 05-10",
                   report=str(args.report.resolve()), catalog=str(member_path.resolve()), n_bins=plots.N_BINS,
                   population_count=population_count, samples_per_trajectory=2*n_plot-1,
                   dense_cache=str(dense_cache), identical_axes_colors_opacity=True, thresholds={})
    cmap = plt.get_cmap("viridis")
    for threshold, overview_ax in zip(thresholds, axes.ravel()):
        indices = np.flatnonzero(maximum >= threshold)
        row_members, row_scores = members[indices], maximum[indices]
        curves = synodic[indices]
        count = len(indices)
        title = (f"Celle con almeno {threshold} asteroidi — unione delle mappe 05–10\n"
                 f"{count:,} traiettorie distinte, {2*n_plot-1} punti ciascuna".replace(",", "."))
        path = output/f"traiettorie_celle_ge_{threshold}_tre_riferimenti.png"
        draw_combined(curves, row_members, row_scores,
                      "Asteroidi nella cella (massimo sulle mappe 05–10)", path, population_count,
                      title=title, normalization=normalization, fixed_alpha=0.12, global_limits=global_limits)
        order = np.argsort(row_scores, kind="stable")
        for start in range(0, count, 2048):
            chunk = order[start:start+2048]
            colors = cmap(normalization(row_scores[chunk]))
            colors[:, 3] = 0.12
            overview_ax.add_collection(LineCollection(curves[chunk].transpose(0, 2, 1), colors=colors,
                                                      linewidths=0.25, rasterized=True))
        d = plots.EARTH_MOON_DISTANCE_KM
        overview_ax.add_patch(Circle((0, 0), plots.R_EARTH_KM, color="#2563eb", zorder=5))
        overview_ax.add_patch(Circle((d, 0), 1737, color="#64748b", zorder=5))
        overview_ax.annotate("Terra", (0, 0), xytext=(4, 5), textcoords="offset points", fontsize=8)
        overview_ax.annotate("Luna", (d, 0), xytext=(4, 5), textcoords="offset points", fontsize=8)
        overview_ax.set_xlim(center[0]-side/2, center[0]+side/2)
        overview_ax.set_ylim(center[1]-side/2, center[1]+side/2)
        overview_ax.set_aspect("equal")
        overview_ax.set_xlabel("x sinodica geocentrica [km]")
        overview_ax.set_ylabel("y [km]")
        overview_ax.grid(alpha=0.15)
        overview_ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        overview_ax.set_title(f"≥ {threshold} asteroidi per cella\n{count:,} traiettorie".replace(",", "."))
        np.savez_compressed(output/f"selezione_ge_{threshold}.npz", members=row_members,
                            source_ids=row_members[:, 0].astype(np.int64),
                            per_map_counts=per_map_counts[indices], maximum_cell_count=row_scores)
        per_map = np.count_nonzero(per_map_counts[indices] >= threshold, axis=0)
        summary["thresholds"][str(threshold)] = dict(trajectories=count,
                per_map_members=dict(zip(("05", "06", "07", "08", "09", "10"), per_map.tolist())))
        print(f"Salvata {path.name}", flush=True)
    for ax in axes.ravel()[len(thresholds):]:
        ax.axis("off")
    visible_axes = list(axes.ravel()[:len(thresholds)])
    overview.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=cmap), ax=visible_axes,
                      shrink=0.8, label="Asteroidi nella cella (massimo sulle mappe 05–10)")
    overview.suptitle("Soglie assolute sulle celle — unione delle mappe 05–10, senza duplicati", fontsize=17)
    overview.supxlabel("Sistema sinodico; stessa scala spaziale, colori e trasparenza. Ogni traiettoria compare una volta per soglia.", fontsize=11)
    overview.savefig(output/"00_confronto_sinodico_soglie.png", dpi=180)
    plt.close(overview)
    np.savez_compressed(output/"membri_selezionati.npz", members=members,
                        source_ids=members[:, 0].astype(np.int64), directions=directions)
    (output/"riepilogo.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Completata serie in {output}", flush=True)


if __name__ == "__main__":
    main()
