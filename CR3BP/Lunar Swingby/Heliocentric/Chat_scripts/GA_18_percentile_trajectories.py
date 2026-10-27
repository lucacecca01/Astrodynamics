"""Traiettorie P97/P99/P99.5/P99.75 con perigei e griglia, per le mappe 05–10."""

import argparse
import gc
import json
import os
from pathlib import Path
import re

import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

import GA_18_plots as plots
import GA_18_high_occurrence_trajectories as high
from GA_18_combined_trajectories import combined_scores, draw_combined, inertial_positions, trajectory_times

MAPS = (
    ("05_eps_e", "05_migliori_transfer_occorrenze_eps_e", r"$\varepsilon$–$e$"),
    ("06_eps_omega", "06_migliori_transfer_occorrenze_eps_omega", r"$\varepsilon$–$\omega$"),
    ("07_hp_e", "07_migliori_transfer_occorrenze_rp_e", r"$h_p$–$e$"),
    ("08_hp_omega", "08_migliori_transfer_occorrenze_rp_omega", r"$h_p$–$\omega$"),
    ("09_hp_theta", "09_migliori_transfer_occorrenze_rp_theta", r"$h_p$–$\theta$"),
    ("10_eps_theta", "10_migliori_transfer_occorrenze_eps_theta", r"$\varepsilon$–$\theta$"),
)


def prepare(grid_dir, output):
    report = (plots.OUTPUT_DIR / "riepilogo.txt").read_text()
    catalog = Path(re.search(r"^Catalogo: (.+)$", report, re.M)[1])
    stats = plots.load_table(catalog, 14)
    stats = stats[np.argsort(stats[:, 0])]
    members = plots.load_table(catalog.with_name("trajectory_exit_catalog.txt"), 12)
    winner_ids = np.array([int(x) for x in re.findall(r"Miglior trasferimento: cluster (\d+)", report)])
    winner_rows = np.searchsorted(stats[:, 0], winner_ids)
    if not np.array_equal(stats[winner_rows, 0], winner_ids):
        raise ValueError("Cluster vincenti incoerenti col catalogo.")
    candidates, _ = combined_scores(members, stats, winner_rows, catalog, cell_counts=False)
    del members
    eps, ecc = candidates[:, 9], candidates[:, 10]
    hp = -plots.MU_EARTH / (2*eps) * (1-ecc) - plots.R_EARTH_KM
    omega = candidates[:, 11] % 360
    directions = plots.winning_cluster_directions(candidates, catalog.with_name("trajectory_exit_catalog.txt"))
    theta = (directions*omega - plots.MOON_RATE_DEG_DAY*candidates[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    scores = np.empty((len(candidates), 6), dtype=np.int64)
    frequencies = np.bincount(winner_rows, minlength=len(stats))
    rows = np.searchsorted(stats[:, 0], candidates[:, 1])
    summaries = {}
    union = np.zeros(len(candidates), dtype=bool)
    for column, ((key, stem, label), (x, y)) in enumerate(zip(MAPS, coordinates)):
        with np.load(grid_dir / f"{stem}_griglie.npz") as data:
            counts = data["occurrences"]
            ex, ey = data["edges_x"], data["edges_y"]
            nx, ny = counts.shape
            ix = np.clip(np.searchsorted(ex, x, side="right")-1, 0, nx-1)
            iy = np.clip(np.searchsorted(ey, y, side="right")-1, 0, ny-1)
            cell = ix*ny+iy
            pairs = np.unique(cell*len(stats)+rows)
            reconstructed = np.bincount(pairs//len(stats), weights=frequencies[pairs % len(stats)],
                                         minlength=counts.size).reshape(counts.shape)
            if not np.array_equal(reconstructed, counts):
                raise ValueError(f"{key}: le griglie non corrispondono ai vincitori del report corrente.")
            scores[:, column] = counts[ix, iy]
            records = []
            for percentile, threshold, mask in zip(data["percentiles"], data["thresholds"], data["masks"]):
                selected = mask[ix, iy]
                if not np.array_equal(selected, scores[:, column] >= threshold):
                    raise ValueError("Maschera percentile incoerente.")
                union |= selected
                records.append(dict(percentile=float(percentile), threshold=float(threshold),
                                    minimum_count=int(np.ceil(threshold)), trajectories=int(selected.sum()),
                                    selected_cells=int(mask.sum()), occupied_cells=int((counts>0).sum()),
                                    total_cells=int(counts.size)))
            summaries[key] = dict(label=label, grid_stem=stem, maximum=int(counts.max()), levels=records)
        print(f"{key}: {[r['trajectories'] for r in records]} traiettorie", flush=True)
    selected = np.flatnonzero(union)
    selected = selected[np.argsort(candidates[selected, 0])]
    summary = dict(population_count=len(candidates), union_for_cache_only=len(selected),
                   selection="Celle della singola mappa sopra il percentile; nessuna aggregazione tra mappe",
                   samples_per_trajectory=201, maps=summaries)
    np.savez_compressed(output / "membri_selezionati.npz", members=candidates[selected],
                        source_ids=candidates[selected, 0].astype(np.int64), per_map_counts=scores[selected])
    (output / "preparazione.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def package_pdf(paths, output):
    pages = []
    try:
        for path in paths:
            with Image.open(path) as img:
                pages.append(img.convert("RGB"))
        pages[0].save(output, "PDF", save_all=True, append_images=pages[1:], resolution=180,
                      quality=95, subsampling=0)
    finally:
        for page in pages:
            page.close()



_RENDER_DATA = None
PLOT_VERSION = "four_percentiles_small_perigees_selected_original_grids_v2"


def render_map(task):
    column, (key, stem, label), output_root = task
    members, scores, curves, limits, summary, resume, grid_dir = _RENDER_DATA
    output = output_root/key
    output.mkdir(parents=True, exist_ok=True)
    info = summary["maps"][key]
    with np.load(grid_dir / f"{stem}_griglie.npz") as grid:
        grid_data = {name: grid[name] for name in ("occurrences", "edges_x", "edges_y")}
    grid_data["xlabel"] = plots.EPS_LABEL if column in (0, 1, 5) else plots.HP_LABEL
    grid_data["ylabel"] = ("e al perigeo [-]" if column in (0, 2) else
                           r"$\omega$ al perigeo, catalogo [deg]" if column in (1, 3) else plots.THETA_LABEL)
    normalization = Normalize(vmin=0, vmax=info["maximum"])
    pages = []
    for record in info["levels"]:
        percentile = record["percentile"]
        tag = f"P{percentile:g}".replace(".", "_")
        indices = np.flatnonzero(scores[:, column] >= record["threshold"])
        if len(indices) != record["trajectories"]:
            raise ValueError("Numero di traiettorie diverso dalla selezione della griglia.")
        selection = members[indices]
        row_scores = scores[indices, column]
        title = (f"Mappa {key[:2]}: {label} — P{percentile:g}, almeno {record['minimum_count']} asteroidi/cella\n"
                 f"{len(indices):,} traiettorie · {record['selected_cells']}/{record['occupied_cells']} celle occupate "
                 f"({100*record['selected_cells']/record['occupied_cells']:.2f}%) · 201 punti per curva")
        path = output / f"{tag}_tre_riferimenti.png"
        selection_path = output/f"{tag}_selezione.npz"
        reusable = False
        if resume and path.is_file() and selection_path.is_file():
            with np.load(selection_path) as saved:
                reusable = (np.array_equal(saved["source_ids"], selection[:, 0].astype(np.int64))
                            and np.array_equal(saved["occurrences"], row_scores)
                            and float(saved["threshold"]) == record["threshold"]
                            and "plot_version" in saved and str(saved["plot_version"]) == PLOT_VERSION)
            if reusable:
                with Image.open(path) as image:
                    image.verify()
        if not reusable:
            grid_data["threshold"] = record["threshold"]
            draw_combined(curves[indices], selection, row_scores, f"Asteroidi per cella — mappa {key[:2]}",
                          path, summary["population_count"], title=title, normalization=normalization,
                          fixed_alpha=0.12, global_limits=limits, raster_chunks=True,
                          perigee_markers=True, selection_grid=grid_data)
        np.savez_compressed(output/f"{tag}_selezione.npz", source_ids=selection[:, 0].astype(np.int64),
                            cluster_ids=selection[:, 1].astype(np.int64), occurrences=row_scores,
                            percentile=percentile, threshold=record["threshold"], plot_version=PLOT_VERSION)
        pages.append(path)
        print(f"Salvata {key} {tag}: {len(indices)} traiettorie", flush=True)
    package_pdf(pages, output / "quattro_percentili_tre_riferimenti.pdf")
    (output / "riepilogo.json").write_text(json.dumps(info, indent=2)+"\n")
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=plots.OUTPUT_DIR / "traiettorie_percentili_per_mappa")
    parser.add_argument("--grids", type=Path, default=plots.OUTPUT_DIR / "mappe_percentili")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2)-2)))
    parser.add_argument("--render-workers", type=int, default=3)
    parser.add_argument("--render-only", action="store_true", help="Riusa selezioni e cache completate e verificate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = (json.loads((args.output / "riepilogo.json").read_text()) if args.render_only
               else prepare(args.grids, args.output))
    with np.load(args.output / "membri_selezionati.npz") as data:
        members, scores = data["members"], data["per_map_counts"]
    # The previous P95 union is a verified superset of every new selection.
    # Refresh thresholds from full grids, never from the selected trajectories.
    for column, (key, stem, _) in enumerate(MAPS):
        with np.load(args.grids / f"{stem}_griglie.npz") as grid:
            counts = grid["occurrences"]
            occupied = counts > 0
            thresholds = np.percentile(counts[occupied], plots.OCCURRENCE_PERCENTILES)
            if thresholds.min() < summary["maps"][key]["levels"][0]["threshold"]:
                raise ValueError("La cache di selezione non copre il nuovo percentile minimo.")
            summary["maps"][key]["levels"] = [
                dict(percentile=float(p), threshold=float(t), minimum_count=int(np.ceil(t)),
                     trajectories=int((scores[:, column] >= t).sum()),
                     selected_cells=int((occupied & (counts >= t)).sum()),
                     occupied_cells=int(occupied.sum()), total_cells=int(counts.size))
                for p, t in zip(plots.OCCURRENCE_PERCENTILES, thresholds)]
    summary["plot_version"] = PLOT_VERSION
    print(f"Cache comune: {len(members)} traiettorie; le selezioni dei grafici restano separate.", flush=True)
    if args.render_only:
        cache = Path(summary["dense_cache"])
        curves = np.load(cache / "curves.npy", mmap_mode="r")
        done = np.load(cache / "done.npy", mmap_mode="r")
        if (not done.all() or curves.shape != (len(members), 2, 201)
                or not np.array_equal(np.load(cache / "members.npy", mmap_mode="r"), members)):
            raise ValueError("Cache incompleta o diversa dai membri selezionati.")
    else:
        sparse, _, _, _ = high.checked_curves(high.DEFAULT_CACHE, members, allow_missing=True)
        gc.collect()
        curves, _, _, cache = high.dense_curves(high.DEFAULT_CACHE, members, sparse, 101, args.workers,
                                               return_inertial=False)
        del sparse
    summary["dense_cache"] = str(cache)
    (args.output / "riepilogo.json").write_text(json.dumps(summary, indent=2)+"\n")
    low, upper = np.full(2, np.inf), np.full(2, -np.inf)
    for start in range(0, len(members), 2048):
        part = curves[start:start+2048]
        times = trajectory_times(members[start:start+2048, 5:7], 101)
        for frame in (part, inertial_positions(part, times)):
            low = np.minimum(low, frame.min(axis=(0, 2)))
            upper = np.maximum(upper, frame.max(axis=(0, 2)))
    limits = np.array([low, upper])
    global _RENDER_DATA
    _RENDER_DATA = (members, scores, curves, limits, summary, args.render_only, args.grids)
    from multiprocessing import get_context
    tasks = [(column, item, args.output) for column, item in enumerate(MAPS)]
    with get_context("fork").Pool(args.render_workers) as pool:
        page_groups = pool.map(render_map, tasks, chunksize=1)
    all_pages = [page for pages in page_groups for page in pages]
    package_pdf(all_pages, args.output / "tutte_le_mappe_quattro_percentili.pdf")
    print(f"Completate {len(all_pages)} figure in {args.output}", flush=True)


if __name__ == "__main__":
    main()
