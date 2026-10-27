"""One figure showing the same selected trajectories in three reference frames."""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

import GA_18_plots as plots


def trajectory_times(bounds_days, n_plot):
    bounds = np.asarray(bounds_days) * 86400 / plots.TIME_UNIT_S
    return np.concatenate((np.linspace(bounds[:, 0], 0, n_plot, axis=1),
                           np.linspace(0, bounds[:, 1], n_plot, axis=1)[:, 1:]), axis=1)


def inertial_positions(synodic_geocentric, times, lunar_origin=False):
    """Rotate axes at each sample; optionally translate to the moving Moon.

    Moon in rotating geocentric axes: (d, 0).
    Moon in inertial geocentric axes: (d cos(t), d sin(t)).
    Thus lunar inertial coordinates are R(t) @ (r_synodic - [d, 0]).
    """
    x = synodic_geocentric[:, 0]
    y = synodic_geocentric[:, 1]
    if lunar_origin:
        x = x - plots.EARTH_MOON_DISTANCE_KM
    cosine, sine = np.cos(times), np.sin(times)
    out = np.empty_like(synodic_geocentric)
    out[:, 0] = cosine*x - sine*y
    out[:, 1] = sine*x + cosine*y
    return out


def draw_combined(synodic, members, scores, score_label, output, population_count,
                  percentage=10, chunk_size=2048, title=None, normalization=None,
                  fixed_alpha=None, global_limits=None, raster_chunks=False,
                  perigee_markers=False, selection_grid=None, return_figure=False, line_width=0.25,
                  opacity_range=None, color_opacities=None):
    """All selected polylines in three panels; same score color in each panel."""
    n_plot = (synodic.shape[-1]+1)//2
    if selection_grid is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), layout="constrained")
    else:
        fig = plt.figure(figsize=(20, 13), layout="constrained")
        layout = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])
        axes = [fig.add_subplot(layout[0, 2*j:2*j+2]) for j in range(3)]
        grid_ax = fig.add_subplot(layout[1, :3])
        original_grid_ax = fig.add_subplot(layout[1, 3:], sharex=grid_ax, sharey=grid_ax)
    if normalization is None:
        normalization = Normalize(vmin=int(np.min(scores)), vmax=max(int(np.max(scores)), int(np.min(scores))+1))
    cmap = plt.get_cmap("viridis")
    limits = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
    renderers = None
    if raster_chunks:
        from matplotlib.backends.backend_agg import RendererAgg
        from matplotlib.transforms import Affine2D, Bbox

        if global_limits is None:
            raise ValueError("Il disegno per blocchi richiede limiti spaziali prefissati.")
        raster_center = np.asarray(global_limits).mean(axis=0)
        raster_side = np.max(np.diff(global_limits, axis=0))*1.06
        full_extent = (raster_center[0]-raster_side/2, raster_center[0]+raster_side/2,
                       raster_center[1]-raster_side/2, raster_center[1]+raster_side/2)
        extents = (full_extent, full_extent, (-100000, 100000, -100000, 100000))
        renderers = [RendererAgg(1200, 1200, 180) for _ in axes]
        transforms = [Affine2D().translate(-left, -bottom).scale(1200/(right-left), 1200/(top-bottom))
                      for left, right, bottom, top in extents]
    # Higher-score curves are drawn last so the dense background does not hide them.
    draw_order = np.argsort(scores, kind="stable")
    for start in range(0, len(members), chunk_size):
        stop = min(start+chunk_size, len(members))
        indices = draw_order[start:stop]
        times = trajectory_times(members[indices, 5:7], n_plot)
        chunk = synodic[indices]
        frames = (chunk, inertial_positions(chunk, times),
                  inertial_positions(chunk, times, lunar_origin=True))
        colors = cmap(normalization(scores[indices]))
        colors[:, 3] = (0.004 + 0.296*normalization(scores[indices])
                        if fixed_alpha is None else fixed_alpha)
        if fixed_alpha is None and opacity_range is not None:
            low_alpha, high_alpha = opacity_range
            colors[:, 3] = low_alpha + (high_alpha-low_alpha)*np.clip(normalization(scores[indices]),0,1)
        if color_opacities is not None:
            from matplotlib.colors import rgb_to_hsv
            hue = rgb_to_hsv(colors[:, :3])[:, 0]
            # Viridis: violet > 240 degrees, blue/green in between, yellow <= 90 degrees.
            colors[:, 3] = np.select((hue >= 2/3, hue <= 1/4),
                                      (color_opacities[0], color_opacities[2]),
                                      default=color_opacities[1])
        for j, (ax, curve) in enumerate(zip(axes, frames)):
            collection = LineCollection(curve.transpose(0, 2, 1), colors=colors,
                                        linewidths=line_width, rasterized=True)
            if renderers is None:
                ax.add_collection(collection)
            else:
                collection.set_transform(transforms[j])
                collection.set_clip_box(Bbox.from_bounds(0, 0, 1200, 1200))
                collection.set_figure(fig)
                collection.draw(renderers[j])
            if j < 2 and global_limits is None:
                points = curve.transpose(0, 2, 1).reshape(-1, 2)
                limits[0] = np.minimum(limits[0], points.min(axis=0))
                limits[1] = np.maximum(limits[1], points.max(axis=0))
    if global_limits is not None:
        limits = np.asarray(global_limits)
    if renderers is not None:
        for ax, renderer, extent in zip(axes, renderers, extents):
            ax.imshow(np.asarray(renderer.buffer_rgba()).copy(), extent=extent,
                      origin="upper", interpolation="none", zorder=1)
    if perigee_markers and len(members):
        # Sample zero is the saved backward perigee, not the ETD midpoint.
        perigees = synodic[:, :, :1]
        peri_times = members[:, 5:6] * 86400 / plots.TIME_UNIT_S
        peri_frames = (perigees, inertial_positions(perigees, peri_times),
                       inertial_positions(perigees, peri_times, lunar_origin=True))
        marker_colors = cmap(normalization(scores))
        for ax, points in zip(axes, peri_frames):
            ax.scatter(points[draw_order, 0, 0], points[draw_order, 1, 0],
                       s=6, facecolors=marker_colors[draw_order], edgecolors="black",
                       linewidths=0.25, alpha=0.95, zorder=4, rasterized=True)
    if selection_grid is not None:
        from matplotlib.colors import ListedColormap
        counts = selection_grid["occurrences"]
        selected_counts = selection_grid.get("selected_occurrences", counts)
        selected = ((selected_counts > 0) if "selected_occurrences" in selection_grid
                    else (counts > 0) & (counts >= selection_grid["threshold"]))
        grid_normalization = selection_grid.get("normalization", normalization)
        grid_ax.pcolormesh(selection_grid["edges_x"], selection_grid["edges_y"],
                           np.ma.masked_where(counts.T <= 0, np.ones(counts.T.shape)),
                           cmap=ListedColormap(["#e3e5e8"]), vmin=0, vmax=1, shading="flat")
        grid_ax.pcolormesh(selection_grid["edges_x"], selection_grid["edges_y"],
                           np.ma.masked_where(~selected.T, selected_counts.T), cmap=cmap,
                           norm=grid_normalization, shading="flat")
        grid_ax.set(xlabel=selection_grid["xlabel"], ylabel=selection_grid["ylabel"],
                    title=(selection_grid.get("selection_title", "Celle selezionate") +
                           f": {selected.sum()}/{np.count_nonzero(counts)} occupate\n"
                           f"({100*selected.sum()/max(1,np.count_nonzero(counts)):.2f}%) · "
                           f"{100*selected.sum()/counts.size:.2f}% della griglia totale"))
        grid_ax.grid(alpha=0.15)
        grid_ax.text(0.01, 0.97, selection_grid.get("selection_caption",
                     "Colore: celle selezionate · grigio: altre celle occupate · bianco: celle vuote"),
                     transform=grid_ax.transAxes, va="top", fontsize=8,
                     bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
        original_grid_ax.pcolormesh(selection_grid["edges_x"], selection_grid["edges_y"],
                                    np.ma.masked_where(counts.T <= 0, counts.T),
                                    cmap=cmap, norm=grid_normalization, shading="flat")
        original_grid_ax.set(xlabel=selection_grid["xlabel"], ylabel=selection_grid["ylabel"],
                             title="Griglia originale completa\nStessi assi e stessa scala colore della selezione")
        original_grid_ax.grid(alpha=0.15)
        if "points" in selection_grid:
            points = np.asarray(selection_grid["points"])
            for panel in (grid_ax, original_grid_ax):
                if len(points):
                    panel.scatter(points[:, 0], points[:, 1], s=18, facecolors="white",
                                  edgecolors="black", linewidths=0.6, zorder=4)
    center = limits.mean(axis=0)
    side = np.max(limits[1]-limits[0])*1.06
    d = plots.EARTH_MOON_DISTANCE_KM
    moon_soi = (plots.MOON_MASS_KG / 5.974e24)**0.4*d
    titles = ("Sinodico Terra–Luna", "Geocentrico inerziale", "Lunocentrico, assi inerziali — zoom")
    for j, (ax, frame_title) in enumerate(zip(axes, titles)):
        ax.set_aspect("equal")
        ax.set_title(frame_title, fontsize=14)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.grid(alpha=0.15)
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        if j < 2:
            ax.add_patch(Circle((0, 0), plots.R_EARTH_KM, color="#2563eb", zorder=5))
            ax.annotate("Terra", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=9)
            ax.set_xlim(center[0]-side/2, center[0]+side/2)
            ax.set_ylim(center[1]-side/2, center[1]+side/2)
        if j == 0:
            ax.add_patch(Circle((d, 0), 1737, color="#64748b", zorder=5))
            ax.annotate("Luna", (d, 0), xytext=(5, 5), textcoords="offset points", fontsize=9)
            ax.add_patch(Circle((d, 0), moon_soi, fill=False, ls="--", color="#888888", lw=0.8))
        elif j == 1:
            ax.add_patch(Circle((0, 0), d, fill=False, ls="--", color="#888888", lw=0.8))
        else:
            ax.add_patch(Circle((0, 0), 1737, color="#64748b", zorder=5))
            ax.annotate("Luna", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=9)
            ax.add_patch(Circle((0, 0), moon_soi, fill=False, ls="--", color="#888888", lw=0.8))
            ax.set_xlim(-100000, 100000)
            ax.set_ylim(-100000, 100000)
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap=cmap)
    separate_grid_scale = selection_grid is not None and "normalization" in selection_grid
    colorbar_axes = (list(axes) if selection_grid is None or separate_grid_scale
                     else list(axes) + [grid_ax, original_grid_ax])
    fig.colorbar(scalar, ax=colorbar_axes, shrink=0.72, label=score_label)
    if separate_grid_scale:
        fig.colorbar(plt.cm.ScalarMappable(norm=grid_normalization, cmap=cmap),
                     ax=[grid_ax, original_grid_ax], shrink=0.85,
                     label=selection_grid.get("colorbar_label", "Asteroidi distinti nella cella"))
    fig.suptitle(title if title is not None else f"{percentage:g}% delle traiettorie con più occorrenze — selezione unica\n"
                 f"{len(members):,} traiettorie su {population_count:,}, {synodic.shape[-1]} punti ciascuna".replace(",", "."),
                 fontsize=16)
    fig.supxlabel(("Pallini con bordo nero: perigei salvati, al tempo del perigeo in ciascun riferimento.\n" if perigee_markers else "") +
                  "Stesse traiettorie e stessi colori nelle tre viste, dal perigeo salvato all’uscita dalla SOI terrestre.\n"
                  "Assi inerziali allineati al sinodico a ETD; nel lunocentrico si sottrae la posizione della Luna a ogni epoca.", fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if not return_figure:
        plt.close(fig)
    return fig if return_figure else None


def combined_scores(members, stats, winner_rows, catalog_path, only_bound=True, n_bins=100, cell_counts=True):
    """Per-source cell counts in all six maps, with identical original bin edges."""
    eps, ecc = members[:, 9], members[:, 10]
    keep = members[:, 1] >= 0
    if only_bound:
        bound = (eps < 0) & (ecc >= 0) & (ecc < 1)
        rp = np.full(len(members), np.nan)
        rp[bound] = -plots.MU_EARTH/(2*eps[bound])*(1-ecc[bound])
        keep &= bound & (rp > plots.R_EARTH_KM)
    keep &= np.isin(members[:, 1], stats[np.unique(winner_rows), 0])
    candidates = members[keep]
    ids = candidates[:, 0].astype(np.int64)
    if not cell_counts:
        return candidates, np.empty((len(candidates), 0), dtype=np.int64)
    scores = np.empty((len(candidates), 6), dtype=np.int64)
    for j, variable in enumerate(("e", "omega")):
        geometry = plots.prepare_parking_geometry(members, stats, only_bound, n_bins, variable)
        data = plots.best_transfer_occurrences(geometry, winner_rows, len(stats))
        x = candidates[:, 9]
        y = candidates[:, 10] if variable == "e" else candidates[:, 11] % 360
        ix = np.clip(np.searchsorted(data["edges_x"], x, side="right")-1, 0, n_bins-1)
        iy = np.clip(np.searchsorted(data["edges_y"], y, side="right")-1, 0, n_bins-1)
        scores[:, j] = data["occurrences"][ix, iy]
    additional = plots.additional_best_transfer_maps(
        members, stats, winner_rows, catalog_path, only_bound, n_bins,
        return_member_occurrences=True)
    for j, (data, *_) in enumerate(additional, start=2):
        source_ids = data["member_source_ids"]
        if not np.array_equal(source_ids, ids):
            raise ValueError("Le sei proiezioni non hanno gli stessi membri ammissibili.")
        scores[:, j] = data["member_occurrences"]
    return candidates, scores


def select_top_sources(members, per_map_scores, percentage, method="max", cluster_frequency=None):
    if method == "max":
        score = per_map_scores.max(axis=1)
    elif method == "sum":
        score = per_map_scores.sum(axis=1)
    elif method == "clusters":
        score = np.array([cluster_frequency[int(c)] for c in members[:, 1]], dtype=np.int64)
    else:
        raise ValueError("Punteggio non riconosciuto.")
    count = int(np.ceil(percentage/100*len(members)))
    order = np.lexsort((members[:, 0], -score))[:count]
    # Restore source-ID order for cache lookup; score does not set drawing order.
    order = order[np.argsort(members[order, 0])]
    return order, score


def main():
    import argparse
    import hashlib
    import json
    import os
    import re
    import GA_18_high_occurrence_trajectories as high

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("max", "sum", "clusters"), default="clusters")
    parser.add_argument("--top-percent", type=float, default=10)
    parser.add_argument("--points-per-segment", type=int, default=101)
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2)-2)))
    parser.add_argument("--report", type=Path, default=plots.OUTPUT_DIR/"riepilogo.txt")
    parser.add_argument("--cache", type=Path, default=high.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.top_percent <= 100 or args.points_per_segment < 101 or args.workers < 1:
        parser.error("Percentuale in (0,100], almeno 101 punti per tratto e almeno un processo.")
    report = args.report.read_text()
    stats_path = Path(re.search(r"^Catalogo: (.+)$", report, re.M)[1])
    member_path = stats_path.with_name("trajectory_exit_catalog.txt")
    stats = plots.load_table(stats_path, 14)
    stats = stats[np.argsort(stats[:, 0])]
    print("Caricamento catalogo e graduatoria unica...", flush=True)
    members = plots.load_table(member_path, 12)
    # Recover row snapshots for older resumable caches. The content digest
    # must reproduce the original cache key before any snapshot is accepted.
    for metadata_path in (high.WORKSPACE/".cache/ga18_high_occurrence_dense").glob("*/metadata.json"):
        snapshot_path = metadata_path.with_name("members.npy")
        if snapshot_path.exists():
            continue
        cached_ids = np.load(metadata_path.with_name("source_ids.npy"), mmap_mode="r")
        indices = np.searchsorted(members[:, 0], cached_ids)
        if np.any(indices >= len(members)) or not np.array_equal(members[indices, 0], cached_ids):
            continue
        snapshot = members[indices]
        digest = hashlib.sha256(json.dumps(json.loads(metadata_path.read_text()), sort_keys=True).encode())
        digest.update(np.ascontiguousarray(snapshot).tobytes())
        if digest.hexdigest()[:20] == metadata_path.parent.name:
            np.save(snapshot_path, snapshot)
    winner_ids = np.array([int(x) for x in re.findall(r"Miglior trasferimento: cluster (\d+)", report)], dtype=int)
    winner_rows = np.searchsorted(stats[:, 0], winner_ids)
    if not np.array_equal(stats[winner_rows, 0], winner_ids):
        raise ValueError("Cluster vincenti assenti dal catalogo.")
    only_bound = "Orbita di parcheggio: legata, perigeo sopra la superficie" in report
    candidates, cell_scores = combined_scores(members, stats, winner_rows, member_path, only_bound, plots.N_BINS,
                                               cell_counts=args.method != "clusters")
    del members
    clusters, counts = np.unique(winner_ids, return_counts=True)
    selected, all_scores = select_top_sources(candidates, cell_scores, args.top_percent, args.method,
                                              dict(zip(clusters, counts)))
    population_count = len(candidates)
    members = candidates[selected]
    scores = all_scores[selected]
    selected_cell_scores = cell_scores[selected]
    cutoff = int(scores.min())
    ties_total = int((all_scores == cutoff).sum())
    ties_selected = int((scores == cutoff).sum())
    print(f"Selezione unica: {len(members)}/{population_count} traiettorie; "
          f"punteggio {cutoff}–{scores.max()}; parità per source_id.", flush=True)
    del candidates, cell_scores, all_scores
    sparse, sparse_inertial, _, old_n_plot = high.checked_curves(args.cache, members, allow_missing=True)
    del sparse_inertial
    n_plot = max(args.points_per_segment, 2*old_n_plot)
    synodic, inertial, directions, dense_cache = high.dense_curves(args.cache, members, sparse, n_plot, args.workers)
    del sparse, inertial
    output = args.output or args.report.parent/f"traiettorie_combinate_top_{args.top_percent:g}_percent_{args.method}_{2*n_plot-1}_punti"
    output.mkdir(parents=True, exist_ok=True)
    labels = {"max": "Massimo delle occorrenze nelle sei celle (05–10)",
              "sum": "Somma delle occorrenze nelle sei celle (05–10)",
              "clusters": "Asteroidi che scelgono il cluster della traiettoria"}
    figure_path = output/"traiettorie_combinate_tre_riferimenti.png"
    draw_combined(synodic, members, scores, labels[args.method], figure_path, population_count, args.top_percent)
    np.savez_compressed(output/"membri_selezionati.npz", members=members, source_ids=members[:, 0].astype(np.int64),
                        scores=scores, per_map_scores=selected_cell_scores, directions=directions)
    np.savetxt(output/"traiettorie_selezionate.txt", np.column_stack((members[:, :2], scores, selected_cell_scores)),
               fmt="%d", header="source_id cluster_id score" + (" occ_05 occ_06 occ_07 occ_08 occ_09 occ_10"
                                                     if selected_cell_scores.shape[1] else ""))
    summary = dict(method=args.method, definition=labels[args.method], percentage=args.top_percent,
                   population_count=population_count, selected_count=len(members), cutoff=cutoff,
                   maximum=int(scores.max()), ties_at_cutoff=ties_total, selected_ties=ties_selected,
                   tie_break="source_id crescente", samples_per_trajectory=2*n_plot-1,
                   report=str(args.report.resolve()), catalog=str(member_path.resolve()),
                   dense_cache=str(dense_cache), frames=["sinodico", "geocentrico inerziale", "lunocentrico con assi inerziali"],
                   lunar_zoom_km=[-100000, 100000])
    (output/"riepilogo.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Salvata figura unica: {figure_path}", flush=True)


if __name__ == "__main__":
    main()
