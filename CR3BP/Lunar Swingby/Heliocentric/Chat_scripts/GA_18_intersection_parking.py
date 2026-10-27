"""Intersezione: >100 asteroidi vincenti per cluster AND soglie su tutte le sei mappe."""

import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize

import GA_18_plots as plots
from GA_18_combined_trajectories import draw_combined, inertial_positions, trajectory_times
from GA_18_percentile_trajectories import MAPS, package_pdf

_RENDER_DATA = None


def render_map(task):
    column, (key, stem, label), output = task
    members, scores, curves, summary, limits, flags, coordinates, grids = _RENDER_DATA
    destination = output/key
    destination.mkdir(parents=True, exist_ok=True)
    counts, ex, ey = grids[column]
    grid = dict(occurrences=counts, edges_x=ex, edges_y=ey,
                xlabel=plots.EPS_LABEL if column in (0, 1, 5) else plots.HP_LABEL,
                ylabel="e al perigeo [-]" if column in (0, 2) else
                r"$\omega$ al perigeo, catalogo [deg]" if column in (1, 3) else plots.THETA_LABEL,
                normalization=Normalize(0, int(counts.max())),
                selection_title="Celle sopra soglia nella proiezione",
                selection_caption="Colore: celle sopra soglia · punti bianchi: orbite nell’intersezione delle sei mappe")
    pages = []
    for index, rec in enumerate(summary["levels"]):
        selected = np.flatnonzero(flags[index].all(axis=1))
        grid["threshold"] = rec["cell_thresholds"][column]
        grid["points"] = coordinates[column][selected]
        tag = f"P{rec['percentile']:g}".replace(".", "_")
        path = destination/f"{tag}_intersezione.png"
        title = (f"Intersezione: >100 asteroidi vincenti per traiettoria e P{rec['percentile']:g} in tutte le sei mappe\n"
                 f"{len(selected)} traiettorie · {rec['clusters']} cluster · "
                 f"{rec['distinct_optimal_asteroids']} asteroidi distinti · proiezione {label}")
        if not len(selected):
            title += " — INTERSEZIONE VUOTA"
        draw_combined(curves[selected], members[selected], scores[selected],
                      "Asteroidi il cui minimo ΔV sceglie il cluster della traiettoria", path,
                      summary["baseline_count"], title=title, normalization=Normalize(0, int(scores.max())),
                      fixed_alpha=0.95, line_width=1.0, global_limits=limits, raster_chunks=True,
                      perigee_markers=True, selection_grid=grid)
        pages.append(path)
    package_pdf(pages, destination/"intersezioni_tutte_le_soglie.pdf")
    print(f"Salvata intersezione {key}", flush=True)
    return pages


def main():
    output = plots.OUTPUT_DIR/"intersezione_orbite_parcheggio"
    output.mkdir(parents=True, exist_ok=True)
    source = plots.OUTPUT_DIR/"traiettorie_percentile_individuale_99_99"
    parent = json.loads((source/"riepilogo.json").read_text())
    with np.load(source/"membri_selezionati.npz") as data:
        keep = data["scores"] > 100
        members, scores = data["members"][keep], data["scores"][keep]
    ids = members[:, 0].astype(np.int64)
    clusters = members[:, 1].astype(np.int64)
    cache = Path(parent["dense_cache"])
    cached_ids = np.load(cache/"source_ids.npy", mmap_mode="r")
    positions = np.searchsorted(cached_ids, ids)
    if not np.array_equal(cached_ids[positions], ids) or not np.load(cache/"done.npy")[positions].all():
        raise ValueError("Cache delle candidate incompleta.")
    if not np.array_equal(np.load(cache/"members.npy", mmap_mode="r")[positions], members):
        raise ValueError("Cache diversa dal catalogo delle candidate.")
    curves = np.load(cache/"curves.npy", mmap_mode="r")[positions]
    direction = np.load(cache/"features.npy", mmap_mode="r")[positions, 3]
    eps, ecc = members[:, 9], members[:, 10]
    hp = -plots.MU_EARTH/(2*eps)*(1-ecc)-plots.R_EARTH_KM
    omega = members[:, 11] % 360
    theta = (direction*omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = [np.column_stack(pair) for pair in
                   ((eps,ecc),(eps,omega),(hp,ecc),(hp,omega),(hp,theta),(eps,theta))]
    percentiles = plots.OCCURRENCE_PERCENTILES
    flags = np.zeros((len(percentiles),len(members),6), dtype=bool)
    cell_scores = np.empty((len(members),6), dtype=np.int64)
    ranks = np.empty((len(members),6))
    thresholds = np.empty((len(percentiles),6))
    grids = []
    for column,(key,stem,_) in enumerate(MAPS):
        with np.load(plots.OUTPUT_DIR/"mappe_percentili"/f"{stem}_griglie.npz") as data:
            counts, ex, ey = data["occurrences"], data["edges_x"], data["edges_y"]
        grids.append((counts, ex, ey))
        xy = coordinates[column]
        ix = np.clip(np.searchsorted(ex,xy[:,0],side="right")-1,0,counts.shape[0]-1)
        iy = np.clip(np.searchsorted(ey,xy[:,1],side="right")-1,0,counts.shape[1]-1)
        cell_scores[:,column] = counts[ix,iy]
        occupied = np.sort(counts[counts>0])
        ranks[:,column] = 100*np.searchsorted(occupied,cell_scores[:,column],side="right")/len(occupied)
        thresholds[:,column] = np.percentile(occupied,percentiles)
        for index, threshold in enumerate(thresholds[:,column]):
            flags[index,:,column] = cell_scores[:,column] >= threshold
            tag = f"P{percentiles[index]:g}".replace(".","_")
            with np.load(plots.OUTPUT_DIR/"traiettorie_percentili_per_mappa"/key/f"{tag}_selezione.npz") as saved:
                if not np.array_equal(flags[index,:,column],np.isin(ids,saved["source_ids"])):
                    raise ValueError("Intersezione diversa dalla selezione originale delle celle.")
    frequency = {int(c):int(scores[clusters==c][0]) for c in np.unique(clusters)}
    levels = []
    for index,p in enumerate(percentiles):
        selected = flags[index].all(axis=1)
        chosen_clusters = np.unique(clusters[selected])
        tag = f"P{p:g}".replace(".","_")
        rec = dict(percentile=float(p), trajectories=int(selected.sum()), clusters=len(chosen_clusters),
                   members_by_cluster={str(c):int(np.count_nonzero(clusters[selected]==c)) for c in chosen_clusters},
                   distinct_optimal_asteroids=sum(frequency[c] for c in chosen_clusters),
                   per_map_pass_counts=flags[index].sum(axis=0).tolist(), cell_thresholds=thresholds[index].tolist())
        levels.append(rec)
        np.savez_compressed(output/f"{tag}_selezione.npz",source_ids=ids[selected],members=members[selected],
                            scores=scores[selected],cell_scores=cell_scores[selected],cell_percentile_ranks=ranks[selected])
        fields = ["source_id","cluster_id","optimal_asteroids","hp_km","eps_km2_s2","e","omega_deg",
                  "theta_deg","t_perigee_days","sense","minimum_cell_percentile"]
        fields += [f"cell_count_{key}" for key,_,_ in MAPS]
        order = np.lexsort((ids,-ranks.min(axis=1),-scores))
        with (output/f"{tag}_candidate.csv").open("w",newline="") as stream:
            writer=csv.writer(stream); writer.writerow(fields)
            for i in order[selected[order]]:
                writer.writerow([ids[i],clusters[i],scores[i],hp[i],eps[i],ecc[i],omega[i],theta[i],
                                 members[i,5],int(direction[i]),ranks[i].min(),*cell_scores[i]])
        print(f"P{p:g}: {rec['trajectories']} orbite, cluster {rec['members_by_cluster']}",flush=True)
    summary=dict(baseline_count=len(members),baseline_rule="Punteggio individuale > 100 asteroidi vincenti",
                 cell_rule="AND delle sei mappe, separatamente per ciascun percentile",levels=levels,
                 model="Prestazioni del modello medio del cluster ereditate dai membri; nessuna nuova verifica Lambert individuale",
                 rank="Ordine descrittivo: punteggio individuale, minimo rango percentile nelle sei mappe, source_id; non ottimo globale")
    (output/"riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"audit_baseline.npz",members=members,source_ids=ids,scores=scores,
                        flags=flags,cell_scores=cell_scores,cell_percentile_ranks=ranks,thresholds=thresholds,
                        percentiles=percentiles)
    times=trajectory_times(members[:,5:7],101)
    inertial=inertial_positions(curves,times)
    limits=np.array([np.minimum(curves.min(axis=(0,2)),inertial.min(axis=(0,2))),
                     np.maximum(curves.max(axis=(0,2)),inertial.max(axis=(0,2)))])
    global _RENDER_DATA
    _RENDER_DATA=(members,scores,curves,summary,limits,flags,coordinates,grids)
    with get_context("fork").Pool(3) as pool:
        page_groups=pool.map(render_map,[(j,item,output) for j,item in enumerate(MAPS)])
    package_pdf([page for group in page_groups for page in group],output/"intersezione_tutte_le_mappe.pdf")
    strongest = next((r for r in reversed(levels) if r["trajectories"]), None)
    if strongest is not None:
        tag = f"P{strongest['percentile']:g}".replace(".","_")
        package_pdf([output/key/f"{tag}_intersezione.png" for key,_,_ in MAPS],
                    output/f"candidate_{tag}.pdf")
    lines=["Intersezione delle orbite di parcheggio", "", ">100 asteroidi vincenti per cluster AND soglia su tutte le sei mappe.", ""]
    lines += [f"P{r['percentile']:g}: {r['trajectories']} traiettorie, {r['clusters']} cluster, "
              f"{r['distinct_optimal_asteroids']} asteroidi distinti." for r in levels]
    if strongest is not None:
        lines += ["", f"La più restrittiva non vuota è P{strongest['percentile']:g}: "
                  f"{strongest['trajectories']} membri, per cluster {strongest['members_by_cluster']}."]
    lines += ["Le candidate ereditano le prestazioni del modello medio del cluster. Occorre una verifica individuale per certificarne il ΔV."]
    (output/"risultati.txt").write_text("\n".join(lines)+"\n")
    print(f"Salvati grafici, tabelle e audit in {output}",flush=True)


if __name__=="__main__":
    main()
