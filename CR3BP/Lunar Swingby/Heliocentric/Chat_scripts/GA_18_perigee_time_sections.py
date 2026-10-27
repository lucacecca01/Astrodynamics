"""Variabilità dei tempi di perigeo in fasce orizzontali dello zoom inerziale."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

import GA_18_plots as plots
from GA_18_combined_trajectories import inertial_positions, trajectory_times


def summarize_sections(y, times, ymin, ymax, number):
    """Fascia 1 in alto; ogni punto assegnato esattamente una volta."""
    height = (ymax-ymin)/number
    if np.any((y<ymin)|(y>ymax)):
        raise ValueError("Perigei fuori dai limiti verticali dello zoom.")
    section = np.clip(np.floor((ymax-y)/height).astype(int),0,number-1)
    records = []
    for index in range(number):
        values = times[section==index]
        records.append(dict(section=index+1,y_min_km=ymax-(index+1)*height,
                            y_max_km=ymax-index*height,count=len(values),
                            t_min_days=float(values.min()) if len(values) else None,
                            t_max_days=float(values.max()) if len(values) else None,
                            span_days=float(np.ptp(values)) if len(values) else None,
                            mean_days=float(values.mean()) if len(values) else None,
                            median_days=float(np.median(values)) if len(values) else None,
                            std_population_days=float(np.std(values,ddof=0)) if len(values) else None))
    return section,records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections",type=int,default=10)
    args = parser.parse_args()
    if args.sections<1:
        parser.error("Il numero di fasce deve essere positivo.")
    root = plots.OUTPUT_DIR/"intersezione_orbite_parcheggio/zoom_arco_geocentrico_inerziale"
    parent = json.loads((root/"riepilogo.json").read_text())
    with np.load(root/"selezione_zoom.npz") as data:
        ids,members,times,points,bounds = (data[k] for k in
            ("source_ids","members","t_perigee_days","perigees_geocentric_inertial_km","bounds_xy_km"))
        coverage = data["coverage"]
        v_soi = data["v_soi_km_s"]
    xmin,xmax,ymin,ymax = bounds
    sections,records = summarize_sections(points[:,1],times,ymin,ymax,args.sections)
    # Recover asteroid identities so repeated trajectories never duplicate coverage.
    cluster_asteroids = {}
    asteroid_id = None
    for line in (plots.OUTPUT_DIR/"riepilogo.txt").read_text().splitlines():
        asteroid_match = re.search(r"\(SPK-ID: (\d+);",line)
        if asteroid_match:
            asteroid_id = asteroid_match.group(1)
        winner_match = re.match(r"\s+Miglior trasferimento: cluster (\d+)",line)
        if winner_match:
            if asteroid_id is None:
                raise ValueError("Cluster vincente senza identificativo asteroide.")
            cluster_asteroids.setdefault(int(winner_match.group(1)),set()).add(asteroid_id)
            asteroid_id = None
    for member,score in zip(members,coverage):
        if len(cluster_asteroids.get(int(member[1]),set()))!=score:
            raise ValueError("Copertura del cluster incoerente con gli asteroidi del riepilogo.")
    section_asteroids = {}
    for index,record in enumerate(records):
        mask = sections==index
        distinct = set().union(*(cluster_asteroids[int(cluster)] for cluster in np.unique(members[mask,1])))
        section_asteroids[str(index+1)] = sorted(distinct)
        record["coverage_unique_asteroids"] = len(distinct)
        record["coverage_max"] = int(coverage[mask].max()) if mask.any() else None
        record["v_soi_span_km_s"] = float(np.ptp(v_soi[mask])) if mask.any() else None
    if sum(r["count"] for r in records)!=len(ids):
        raise ValueError("Conteggio delle fasce incoerente.")
    cache = Path(parent["dense_cache"])
    if (not np.array_equal(np.load(cache/"source_ids.npy"),ids)
            or not np.array_equal(np.load(cache/"members.npy"),members)
            or not np.load(cache/"done.npy").all()):
        raise ValueError("Cache dello zoom incoerente.")
    synodic = np.load(cache/"curves.npy",mmap_mode="r")
    curves = inertial_positions(synodic,trajectory_times(members[:,5:7],(synodic.shape[-1]+1)//2))
    output = root/f"tempi_{args.sections}_fasce_orizzontali"
    output.mkdir(parents=True,exist_ok=True)
    (output/"asteroidi_distinti_per_fascia.json").write_text(json.dumps(section_asteroids,indent=2)+"\n")
    with (output/"variabilita_temporale.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (output/"riepilogo_fasce.csv").open("w",newline="") as stream:
        fields = ["section","coverage_max","coverage_unique_asteroids","v_soi_span_km_s","span_days"]
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    with (output/"perigei_per_fascia.csv").open("w",newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["source_id","cluster_id","section","x_km","y_km","t_perigee_days"])
        for sid,member,section,point,t in zip(ids,members,sections,points,times):
            writer.writerow([int(sid),int(member[1]),int(section+1),*point,t])
    height = (ymax-ymin)/args.sections
    norm = Normalize(float(times.min()),float(times.max()))
    # Matplotlib's s is marker area: make area proportional to coverage.
    marker_areas = 36.0*coverage/coverage.max()
    order = np.argsort(times,kind="stable")
    colors = plt.get_cmap("viridis")(norm(times[order])); colors[:,3]=.4
    fig,(ax,table_ax) = plt.subplots(1,2,figsize=(20,11),gridspec_kw={"width_ratios":[1,1.25]},layout="constrained")
    ax.add_collection(LineCollection(curves[order].transpose(0,2,1),colors=colors,linewidths=.6,rasterized=True))
    ax.scatter(points[:,0],points[:,1],c=times,cmap="viridis",norm=norm,s=marker_areas,
               edgecolors="black",linewidths=.4,zorder=4)
    for k in range(args.sections+1):
        ax.axhline(ymax-k*height,color="black",lw=.6,ls="--",alpha=.4)
    for record in records:
        center=(record["y_min_km"]+record["y_max_km"])/2
        if record["span_days"] is not None and record["span_days"]>=28:
            ax.axhspan(record["y_min_km"],record["y_max_km"],color="#f6b940",alpha=.16,zorder=0)
        ax.text(xmax-.025*(xmax-xmin),center,str(record["section"]),ha="right",va="center",fontsize=9,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=1),zorder=5)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel="x geocentrica inerziale [km]",ylabel="y [km]",
           title=f"{len(ids)} perigei · fasce alte {height:,.0f} km".replace(",","."))
    ax.set_aspect("equal"); ax.grid(alpha=.12)
    ax.ticklabel_format(style="sci",scilimits=(0,0),axis="both")
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="viridis"),ax=ax,shrink=.8,
                    label="Tempo del perigeo rispetto a ETD = 0")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g} d"))
    table_ax.axis("off")
    size_handles = [table_ax.scatter([],[],s=36.0*value/coverage.max(),facecolors="gray",
                                    edgecolors="black",linewidths=.4,label=str(int(value)))
                    for value in np.unique(coverage)]
    table_ax.legend(handles=size_handles,title="Area del punto ∝ copertura [asteroidi vincenti del cluster]",
                    loc="lower center",bbox_to_anchor=(.5,.13),ncol=len(size_handles),frameon=False)
    def formatted(value):
        return "—" if value is None else f"{value:.2f}"
    table_rows = [[str(r["section"]),str(r["coverage_max"]) if r["coverage_max"] is not None else "—",
                   str(r["coverage_unique_asteroids"]),
                   f"{r['v_soi_span_km_s']:.4f}" if r["v_soi_span_km_s"] is not None else "—",
                   formatted(r["span_days"])] for r in records]
    table=table_ax.table(cellText=table_rows,colLabels=["Fascia","Copertura max","Copertura cumulativa\n(asteroidi distinti)","ΔV SOI [km/s]","Δt [d]"],
                        cellLoc="center",loc="center",colWidths=[.09,.21,.28,.24,.15])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,2.2)
    for (row,column),cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row==0:
            cell.set_facecolor("#dce8f1"); cell.set_text_props(weight="bold")
        elif row%2==0:
            cell.set_facecolor("#f3f5f7")
        if row>0 and records[row-1]["span_days"] is not None and records[row-1]["span_days"]>=28:
            cell.set_facecolor("#fff0c2"); cell.set_text_props(weight="bold")
    table_ax.set_title("Variabilità tra i perigei di ciascuna fascia",fontsize=14)
    fig.suptitle(f"Tempo del perigeo: zoom diviso in {args.sections} sezioni orizzontali",fontsize=17)
    fig.supxlabel("Fasce numerate dall’alto verso il basso. Copertura max: massimo numero di asteroidi vincenti del cluster di una traiettoria nella fascia.\n"
                  "ΔV SOI e Δt: massimo − minimo delle velocità alla SOI e dei tempi di perigeo nella fascia. Tempi rispetto a ETD = 0.\n"
                  "Copertura cumulativa: unione degli asteroidi vincenti dei cluster presenti nella singola fascia, senza duplicati.\n"
                  "Punti: colore = tempo del perigeo; area proporzionale alla copertura. Statistiche sui soli perigei nella fascia.",fontsize=10)
    fig.savefig(output/f"zoom_{args.sections}_fasce_variabilita.pdf",dpi=200)
    fig.savefig(output/f"zoom_{args.sections}_fasce_variabilita.png",dpi=180)
    plt.close(fig)
    summary=dict(sections=args.sections,section_height_km=height,trajectories=len(ids),bounds_xy_km=bounds.tolist(),
                 numbering="1 in alto, ultima fascia in basso",definition="Tempi dei soli perigei nella fascia, rispetto a ETD",
                 point_edge_color="black",point_area="36 * coverage / max(coverage) [pt²]",
                 std_ddof=0,records=records)
    (output/"riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
    for r in records:
        print(f"{r['section']:2}  N={r['count']:2}  {formatted(r['t_min_days']):>6} … {formatted(r['t_max_days']):>6}  "
              f"Δ={formatted(r['span_days']):>5}  σ={formatted(r['std_population_days']):>5}")
    print(output)


if __name__=="__main__":
    main()
