"""Zoom dell'arco dei perigei: copertura, V_SOI e tempo, in assi geocentrici inerziali."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

import GA_18_plots as plots
import GA_18_high_occurrence_trajectories as high
from GA_18_combined_trajectories import inertial_positions, trajectory_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds",type=float,nargs=4,default=(-140000,100000,-350000,-90000),
                        metavar=("XMIN","XMAX","YMIN","YMAX"))
    args = parser.parse_args()
    xmin,xmax,ymin,ymax = args.bounds
    if not np.isfinite(args.bounds).all() or xmin>=xmax or ymin>=ymax:
        parser.error("I limiti devono essere finiti e ordinati.")
    root = plots.OUTPUT_DIR/"intersezione_orbite_parcheggio"
    with np.load(root/"P97_selezione.npz") as data:
        members,ids,coverage = data["members"],data["source_ids"],data["scores"]
    parent = json.loads((plots.OUTPUT_DIR/"traiettorie_percentile_individuale_99_99"/"riepilogo.json").read_text())
    cache = Path(parent["dense_cache"])
    indices = np.searchsorted(np.load(cache/"source_ids.npy",mmap_mode="r"),ids)
    if (not np.array_equal(np.load(cache/"source_ids.npy",mmap_mode="r")[indices],ids)
            or not np.load(cache/"done.npy",mmap_mode="r")[indices].all()
            or not np.array_equal(np.load(cache/"members.npy",mmap_mode="r")[indices],members)):
        raise ValueError("Cache diversa dalla selezione sorgente.")
    sparse = np.load(cache/"curves.npy",mmap_mode="r")[indices]
    perigees = inertial_positions(sparse[:,:,:1],members[:,5:6]*86400/plots.TIME_UNIT_S)[:,:,0]
    selected = ((perigees[:,0]>=xmin)&(perigees[:,0]<=xmax)&
                (perigees[:,1]>=ymin)&(perigees[:,1]<=ymax))
    if not selected.any():
        raise ValueError("Nessun perigeo nel riquadro.")
    members,ids,coverage = members[selected],ids[selected],coverage[selected]
    synodic,_,_,dense_cache = high.dense_curves(high.DEFAULT_CACHE,members,sparse[selected],201,8,return_inertial=False)
    curves = inertial_positions(synodic,trajectory_times(members[:,5:7],201))
    perigees = curves[:,:,0]
    v_soi = np.linalg.norm(members[:,2:5],axis=1)
    t_perigee = members[:,5]
    output = root/"zoom_arco_geocentrico_inerziale"
    output.mkdir(parents=True,exist_ok=True)
    variants = (
        ("01_copertura",coverage,"Copertura", "Asteroidi il cui minimo ΔV sceglie il cluster della traiettoria"),
        ("02_V_soi",v_soi,r"$V_{\rm SOI}$",r"$V_{\rm SOI}=|\mathbf{v}_{\rm SOI}|$ [km/s]"),
        ("03_t_perigeo",t_perigee,"Tempo del perigeo",r"$t_{\rm perigeo}-t_{\rm ETD}$ [giorni]"),
    )
    overview,overview_axes = plt.subplots(1,3,figsize=(21,8),layout="constrained")
    ranges = {}
    with PdfPages(output/"zoom_copertura_Vsoi_tperigeo.pdf") as pdf:
        for (name,values,title,label),overview_ax in zip(variants,overview_axes):
            minimum,maximum = float(values.min()),float(values.max())
            norm = Normalize(minimum,maximum if maximum>minimum else minimum+1)
            order = np.argsort(values,kind="stable")
            colors = plt.get_cmap("viridis")(norm(values[order]))
            colors[:,3] = .4
            fig,ax = plt.subplots(figsize=(8,8),layout="constrained")
            for target,current_fig in ((ax,fig),(overview_ax,overview)):
                target.add_collection(LineCollection(curves[order].transpose(0,2,1),colors=colors,
                                                     linewidths=.8,rasterized=True))
                target.scatter(perigees[order,0],perigees[order,1],c=values[order],norm=norm,cmap="viridis",
                               s=19,edgecolors="black",linewidths=.45,zorder=4,rasterized=True)
                target.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel="x geocentrica inerziale [km]",ylabel="y [km]")
                target.set_aspect("equal")
                target.ticklabel_format(style="sci",scilimits=(0,0),axis="both")
                target.grid(alpha=.18)
                target.set_title(title,fontsize=14)
                cb = current_fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="viridis"),ax=target,shrink=.82,label=label)
                if name=="03_t_perigeo":
                    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g} d"))
                elif name=="01_copertura":
                    cb.locator = MaxNLocator(nbins=7,integer=True); cb.update_ticks()
            fig.suptitle(f"Arco dei perigei — geocentrico inerziale\n{len(ids)} traiettorie con perigeo nello zoom",fontsize=15)
            fig.supxlabel("Pallini: perigei salvati. Opacità delle curve: 0,4. Scala colore adattata alle traiettorie nello zoom.\n"
                          "Tempi rispetto a ETD = 0; copertura ereditata dal modello medio del cluster.",fontsize=9)
            fig.savefig(output/f"{name}.png",dpi=200)
            fig.savefig(output/f"{name}.pdf",dpi=200)
            pdf.savefig(fig,dpi=200)
            plt.close(fig)
            ranges[name]=[minimum,maximum]
            print(f"{name}: {minimum:.6f} … {maximum:.6f}",flush=True)
    overview.suptitle(f"Zoom dei perigei — {len(ids)} traiettorie, geocentrico inerziale",fontsize=16)
    overview.supxlabel("Stessa selezione e stessi limiti nei tre grafici; scale colore locali. Curve con opacità 0,4 e perigei evidenziati.",fontsize=10)
    overview.savefig(output/"00_confronto_tre_colorazioni.png",dpi=180)
    overview.savefig(output/"00_confronto_tre_colorazioni.pdf",dpi=180)
    plt.close(overview)
    np.savez_compressed(output/"selezione_zoom.npz",source_ids=ids,members=members,coverage=coverage,
                        v_soi_km_s=v_soi,t_perigee_days=t_perigee,perigees_geocentric_inertial_km=perigees,
                        bounds_xy_km=args.bounds)
    (output/"riepilogo.json").write_text(json.dumps(dict(
        source_selection=str(root/"P97_selezione.npz"),bounds_xy_km=args.bounds,trajectories=len(ids),
        clusters=len(np.unique(members[:,1])),selection="Perigeo geocentrico inerziale dentro il riquadro",
        frame="Geocentrico inerziale, assi allineati al sinodico a ETD",alpha=.4,
        samples_per_trajectory=401,dense_cache=str(dense_cache),color_ranges=ranges),indent=2)+"\n")
    print(f"Salvati i tre zoom e il confronto in {output}",flush=True)


if __name__=="__main__":
    main()
