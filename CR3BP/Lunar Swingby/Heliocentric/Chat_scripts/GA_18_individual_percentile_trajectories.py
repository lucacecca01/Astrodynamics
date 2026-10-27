"""Percentili delle singole traiettorie per numero di asteroidi col cluster vincente.

Ogni membro eredita le prestazioni del modello medio del cluster. Le parità
alla soglia sono incluse; una sola selezione per percentile in tutte le mappe.
"""

import argparse
import gc
import json
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np
from matplotlib.colors import Normalize

import GA_18_plots as plots
import GA_18_high_occurrence_trajectories as high
from GA_18_combined_trajectories import combined_scores, draw_combined, inertial_positions, trajectory_times
from GA_18_percentile_trajectories import MAPS, package_pdf

_RENDER_DATA = None


def prepare(output, percentiles=plots.OCCURRENCE_PERCENTILES):
    report_path = plots.OUTPUT_DIR / "riepilogo.txt"
    report = report_path.read_text()
    catalog = Path(re.search(r"^Catalogo: (.+)$", report, re.M)[1])
    stats = plots.load_table(catalog, 14)
    stats = stats[np.argsort(stats[:, 0])]
    all_members = plots.load_table(catalog.with_name("trajectory_exit_catalog.txt"), 12)
    winner_ids = np.array([int(x) for x in re.findall(r"Miglior trasferimento: cluster (\d+)", report)])
    winner_rows = np.searchsorted(stats[:, 0], winner_ids)
    if not np.array_equal(stats[winner_rows, 0], winner_ids):
        raise ValueError("Cluster vincenti assenti dal catalogo.")
    members, _ = combined_scores(all_members, stats, winner_rows, catalog, cell_counts=False)
    del all_members
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    frequency = np.bincount(winner_rows, minlength=len(stats))
    scores = frequency[rows]
    thresholds = np.percentile(scores, percentiles)
    keep = scores >= thresholds.min()
    chosen = np.flatnonzero(keep)
    chosen = chosen[np.argsort(members[chosen, 0])]
    records = []
    for percentile, threshold in zip(percentiles, thresholds):
        selected = scores >= threshold
        selected_rows = np.unique(rows[selected])
        rec = dict(percentile=float(percentile), threshold=float(threshold),
                   minimum_count=int(np.ceil(threshold)), trajectories=int(selected.sum()),
                   trajectory_fraction=float(selected.mean()), clusters=len(selected_rows),
                   distinct_optimal_asteroids=int(frequency[selected_rows].sum()))
        records.append(rec)
        print(f"P{percentile:g}: soglia {rec['minimum_count']}, {rec['trajectories']} traiettorie "
              f"({100*rec['trajectory_fraction']:.3f}%), {rec['clusters']} cluster", flush=True)
    eps, ecc = members[:, 9], members[:, 10]
    hp = -plots.MU_EARTH/(2*eps)*(1-ecc)-plots.R_EARTH_KM
    omega = members[:, 11] % 360
    direction = plots.winning_cluster_directions(members, catalog.with_name("trajectory_exit_catalog.txt"))
    theta = (direction*omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps,ecc),(eps,omega),(hp,ecc),(hp,omega),(hp,theta),(eps,theta))
    for column, ((key,stem,_), (x,y)) in enumerate(zip(MAPS,coordinates)):
        with np.load(plots.OUTPUT_DIR/"mappe_percentili"/f"{stem}_griglie.npz") as grid:
            counts, ex, ey = grid["occurrences"], grid["edges_x"], grid["edges_y"]
            ix = np.clip(np.searchsorted(ex,x,side="right")-1,0,counts.shape[0]-1)
            iy = np.clip(np.searchsorted(ey,y,side="right")-1,0,counts.shape[1]-1)
            cell = ix*counts.shape[1]+iy
            pairs = np.unique(cell*len(stats)+rows)
            reconstructed = np.bincount(pairs//len(stats),weights=frequency[pairs%len(stats)],
                                         minlength=counts.size).reshape(counts.shape)
            if not np.array_equal(reconstructed,counts):
                raise ValueError(f"{key}: vincitori diversi da quelli delle griglie originali.")
            subset_cells, subset_rows = cell[chosen], rows[chosen]
            selected_grids = []
            for rec in records:
                selected = scores[chosen]>=rec["threshold"]
                pairs = np.unique(subset_cells[selected]*len(stats)+subset_rows[selected])
                sub = np.bincount(pairs//len(stats),weights=frequency[pairs%len(stats)],
                                   minlength=counts.size).reshape(counts.shape).astype(np.int64)
                if np.any(sub>counts):
                    raise ValueError("Copertura selezionata maggiore di quella originale.")
                selected_grids.append(sub)
            np.savez_compressed(output/f"{key}_griglie.npz", occurrences=counts, edges_x=ex, edges_y=ey,
                                selected_occurrences=selected_grids, percentiles=percentiles)
    np.savez_compressed(output/"membri_selezionati.npz", members=members[chosen],
                        source_ids=members[chosen,0].astype(np.int64), scores=scores[chosen])
    for rec in records:
        selected = scores[chosen]>=rec["threshold"]
        tag=f"P{rec['percentile']:g}".replace(".","_")
        np.savez_compressed(output/f"{tag}_selezione.npz", source_ids=members[chosen[selected],0].astype(np.int64),
                            cluster_ids=members[chosen[selected],1].astype(np.int64), scores=scores[chosen[selected]],
                            threshold=rec["threshold"], percentile=rec["percentile"])
    summary=dict(report=str(report_path), catalog=str(catalog), population_count=len(members),
                 maximum_individual_score=int(scores.max()), levels=records, samples_per_trajectory=201,
                 score_histogram=np.bincount(scores).tolist(),
                 criterion="Numero di asteroidi il cui minimo DV sceglie il cluster della traiettoria",
                 percentile_population="Tutte le singole traiettorie ammissibili dei cluster vincenti",
                 ties="Incluse tutte le traiettorie con punteggio uguale alla soglia",
                 model="Prestazioni del modello medio ereditate dai membri; non Lambert individuale",
                 grid="Copertura dei soli cluster selezionati, ogni asteroide al massimo una volta per cella")
    (output/"riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def render_map(task):
    column,(key,stem,label),output=task
    members,scores,curves,summary,limits=_RENDER_DATA
    destination=output/key
    destination.mkdir(parents=True,exist_ok=True)
    with np.load(output/f"{key}_griglie.npz") as stored:
        grid={k:stored[k] for k in ("occurrences","edges_x","edges_y")}
        selected_grids=stored["selected_occurrences"]
    grid.update(xlabel=plots.EPS_LABEL if column in (0,1,5) else plots.HP_LABEL,
                ylabel="e al perigeo [-]" if column in (0,2) else
                r"$\omega$ al perigeo, catalogo [deg]" if column in (1,3) else plots.THETA_LABEL,
                normalization=Normalize(0,int(grid["occurrences"].max())),
                selection_title="Celle con traiettorie selezionate",
                selection_caption="Colore: copertura dei cluster selezionati · grigio: altre celle occupate",
                colorbar_label="Asteroidi distinti nella cella — copertura dei cluster")
    pages=[]
    fig=None
    previous_selected=None
    previous_grid=None
    for index,rec in enumerate(summary["levels"]):
        selected=np.flatnonzero(scores>=rec["threshold"])
        tag=f"P{rec['percentile']:g}".replace(".","_")
        grid["selected_occurrences"]=selected_grids[index]
        title=(f"P{rec['percentile']:g} delle singole traiettorie — almeno {rec['minimum_count']} trasferimenti ottimali per cluster\n"
               f"{rec['trajectories']:,} traiettorie ({100*rec['trajectory_fraction']:.2f}%) · "
               f"{rec['clusters']} cluster · {rec['distinct_optimal_asteroids']} asteroidi · proiezione {label}")
        path=destination/f"{tag}_tre_riferimenti.png"
        if (fig is not None and np.array_equal(selected,previous_selected)
                and np.array_equal(selected_grids[index],previous_grid)):
            # Identical selections differ only in their percentile title.
            fig.suptitle(title,fontsize=16)
            fig.savefig(path,dpi=180)
        else:
            if fig is not None:
                plots.plt.close(fig)
            fig=draw_combined(curves[selected],members[selected],scores[selected],
                          "Asteroidi il cui minimo ΔV sceglie il cluster della traiettoria",path,
                          summary["population_count"],title=title,normalization=Normalize(0,summary["maximum_individual_score"]),
                          fixed_alpha=0.12,global_limits=limits,raster_chunks=True,perigee_markers=True,
                          selection_grid=grid,return_figure=True)
        previous_selected=selected
        previous_grid=selected_grids[index]
        pages.append(path)
        print(f"Salvata {key} {tag}",flush=True)
    if fig is not None:
        plots.plt.close(fig)
    package_pdf(pages,destination/"percentili_tre_riferimenti.pdf")
    return pages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=plots.OUTPUT_DIR/"traiettorie_percentili_individuali")
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--render-workers",type=int,default=3)
    parser.add_argument("--percentiles",type=float,nargs="+",default=plots.OCCURRENCE_PERCENTILES)
    args=parser.parse_args()
    if not args.percentiles or any(not np.isfinite(p) or not 0<p<100 for p in args.percentiles):
        parser.error("I percentili devono essere finiti e compresi fra 0 e 100 esclusi.")
    percentiles=sorted(set(args.percentiles))
    args.output.mkdir(parents=True,exist_ok=True)
    summary=prepare(args.output,percentiles)
    with np.load(args.output/"membri_selezionati.npz") as data:
        members,scores=data["members"],data["scores"]
    gc.collect()
    sparse,_,_,_=high.checked_curves(high.DEFAULT_CACHE,members,allow_missing=True)
    curves,_,_,cache=high.dense_curves(high.DEFAULT_CACHE,members,sparse,101,args.workers,return_inertial=False)
    del sparse
    summary["dense_cache"]=str(cache)
    (args.output/"riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
    low,upper=np.full(2,np.inf),np.full(2,-np.inf)
    for start in range(0,len(members),2048):
        part=curves[start:start+2048]
        times=trajectory_times(members[start:start+2048,5:7],101)
        for frame in (part,inertial_positions(part,times)):
            low=np.minimum(low,frame.min(axis=(0,2)))
            upper=np.maximum(upper,frame.max(axis=(0,2)))
    global _RENDER_DATA
    _RENDER_DATA=(members,scores,curves,summary,np.array([low,upper]))
    with get_context("fork").Pool(args.render_workers) as pool:
        pages=pool.map(render_map,[(j,item,args.output) for j,item in enumerate(MAPS)],chunksize=1)
    package_pdf([page for group in pages for page in group],args.output/"tutte_le_mappe_percentili_individuali.pdf")
    print(f"Completate {len(MAPS)*len(percentiles)} figure in {args.output}",flush=True)


if __name__=="__main__":
    main()
