"""Un'unica figura delle traiettorie comuni alle sei mappe a un percentile."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from matplotlib.colors import Normalize

import GA_18_plots as plots
from GA_18_combined_trajectories import draw_combined, inertial_positions, trajectory_times
from GA_18_percentile_trajectories import MAPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percentile",type=float,default=99.5)
    parser.add_argument("--alpha",type=float,default=None)
    parser.add_argument("--line-width",type=float,default=None)
    parser.add_argument("--score-opacity",action="store_true",
                        help="Opacità crescente con il punteggio per far emergere i colori delle traiettorie migliori")
    parser.add_argument("--color-opacities",type=float,nargs=3,metavar=("VIOLA","BLU","GIALLO"),
                        help="Opacità per le fasce viola, blu/verdi e gialle della scala viridis")
    args = parser.parse_args()
    if args.alpha is not None and (not np.isfinite(args.alpha) or not 0<args.alpha<=1):
        parser.error("alpha deve essere in (0, 1].")
    if args.line_width is not None and (not np.isfinite(args.line_width) or args.line_width<=0):
        parser.error("line-width deve essere positivo e finito.")
    if args.color_opacities is not None and any(not np.isfinite(v) or not 0<=v<=1 for v in args.color_opacities):
        parser.error("Le tre opacità devono essere comprese fra 0 e 1.")
    tag = f"P{args.percentile:g}".replace(".","_")
    parent = plots.OUTPUT_DIR/"traiettorie_percentili_per_mappa"
    common = None
    input_counts = {}
    for key,_,_ in MAPS:
        with np.load(parent/key/f"{tag}_selezione.npz") as data:
            ids = data["source_ids"]
            input_counts[key] = len(ids)
            common = ids if common is None else np.intersect1d(common,ids)
    if not len(common):
        raise ValueError("Nessuna traiettoria comune alle sei mappe.")
    summary = json.loads((parent/"riepilogo.json").read_text())
    with np.load(parent/"membri_selezionati.npz") as data:
        indices = np.searchsorted(data["source_ids"],common)
        if not np.array_equal(data["source_ids"][indices],common):
            raise ValueError("Traiettorie comuni assenti dalla selezione sorgente.")
        members = data["members"][indices]
        cell_scores = data["per_map_counts"][indices]
    cache = Path(summary["dense_cache"])
    cached_ids = np.load(cache/"source_ids.npy",mmap_mode="r")
    indices = np.searchsorted(cached_ids,common)
    if (not np.array_equal(cached_ids[indices],common)
            or not np.load(cache/"done.npy",mmap_mode="r")[indices].all()
            or not np.array_equal(np.load(cache/"members.npy",mmap_mode="r")[indices],members)):
        raise ValueError("Cache delle traiettorie comuni incompleta o incoerente.")
    curves = np.load(cache/"curves.npy",mmap_mode="r")[indices]
    report = (plots.OUTPUT_DIR/"riepilogo.txt").read_text()
    frequency = Counter(map(int,re.findall(r"Miglior trasferimento: cluster (\d+)",report)))
    clusters = members[:,1].astype(np.int64)
    scores = np.array([frequency[int(cluster)] for cluster in clusters])
    alpha = args.alpha if args.alpha is not None else (0.12 if len(common)>500 else 0.9)
    if args.score_opacity or args.color_opacities is not None:
        alpha = None
    line_width = args.line_width if args.line_width is not None else (0.25 if len(common)>500 else 0.8)
    times = trajectory_times(members[:,5:7],101)
    inertial = inertial_positions(curves,times)
    limits = np.array([np.minimum(curves.min(axis=(0,2)),inertial.min(axis=(0,2))),
                       np.maximum(curves.max(axis=(0,2)),inertial.max(axis=(0,2)))])
    output = plots.OUTPUT_DIR/f"traiettorie_comuni_sei_mappe_{tag}"
    output.mkdir(parents=True,exist_ok=True)
    title = (f"Traiettorie comuni a tutte le sei mappe — P{args.percentile:g}\n"
             f"{len(common)} traiettorie · {len(np.unique(clusters))} cluster · "
             "201 punti per curva · nessun filtro aggiuntivo sul punteggio individuale")
    fig = draw_combined(curves,members,scores,
                        "Asteroidi il cui minimo ΔV sceglie il cluster della traiettoria",
                        output/"traiettorie_comuni_tre_riferimenti.png",summary["population_count"],
                        title=title,normalization=Normalize(0,int(scores.max())),fixed_alpha=alpha,
                        global_limits=limits,raster_chunks=True,perigee_markers=True,
                        return_figure=True,line_width=line_width,
                        opacity_range=(0.035,0.9) if args.score_opacity else None,
                        color_opacities=args.color_opacities)
    try:
        fig.savefig(output/"traiettorie_comuni_tre_riferimenti.pdf",dpi=180)
    finally:
        plots.plt.close(fig)
    np.savez_compressed(output/"selezione.npz",source_ids=common,members=members,
                        scores=scores,per_map_counts=cell_scores)
    details = dict(percentile=args.percentile,rule="AND delle sei mappe, nessun filtro sul punteggio individuale",
                   trajectories=len(common),clusters=len(np.unique(clusters)),
                   members_by_cluster={str(c):int(np.count_nonzero(clusters==c)) for c in np.unique(clusters)},
                   input_counts=input_counts,trajectories_above_100=int(np.count_nonzero(scores>100)),
                   dense_cache=str(cache),samples_per_trajectory=201,
                   rendering=dict(alpha=alpha,line_width=line_width,perigee_markers=True,
                                  opacity_range=[0.035,0.9] if args.score_opacity else None,
                                  color_opacities=args.color_opacities))
    (output/"riepilogo.json").write_text(json.dumps(details,indent=2)+"\n")
    print(f"Salvata figura unica: {len(common)} traiettorie, {len(np.unique(clusters))} cluster, in {output}")


if __name__=="__main__":
    main()
