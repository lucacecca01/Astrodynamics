"""Disponibilità giornaliera delle famiglie, usando tutte le opportunità salvate.

Le date sono del perigeo del modello medio del cluster, in TDB; i contenitori
giornalieri non certificano la disponibilità a ogni istante del giorno.
"""

from collections import Counter
import csv
import json
from multiprocessing import get_context
from pathlib import Path
import time
import warnings

import numpy as np
from matplotlib import pyplot as plt, dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

import GA_18_plots as plots

_CONFIG = None
FAMILY_NAMES = ("Tutti i cluster ammissibili", "Famiglia comune P97", "Famiglia comune P99,5", "Cluster con >100 vincitori")


def analyze_file(task):
    index, path, reference, spkid = task
    stats, eligible, family_rows = _CONFIG
    table = plots.load_table(path, 5)
    if not len(table):
        return index, spkid, None
    rows = np.searchsorted(stats[:,0], table[:,1])
    if np.any(rows >= len(stats)) or not np.array_equal(stats[rows,0], table[:,1]):
        raise ValueError(f"Cluster sconosciuti in {path}")
    keep = eligible[rows] & plots.transfer_mask(table, stats, plots.MAX_DV_KM_S, plots.MAX_DURATION_DAYS)
    table, rows = table[keep], rows[keep]
    if not len(table):
        return index, spkid, None
    peri = table[:,0]+stats[rows,6]-(stats[rows,7] if reference=="SOI" else 0)
    days, inverse = np.unique(np.floor(peri+0.5).astype(np.int64), return_inverse=True)
    row_days = days[inverse]
    order = np.lexsort((table[:,1],table[:,2],table[:,4],row_days))
    _, first = np.unique(row_days[order], return_index=True)
    best = order[first]
    minima = np.full((len(FAMILY_NAMES), len(days)), np.inf)
    minima[0] = table[best,4]
    for family_index, row_mask in enumerate(family_rows, 1):
        selected = row_mask[rows]
        np.minimum.at(minima[family_index],inverse[selected],table[selected,4])
    # One witness transfer per day for the P97 family; retain its actual time.
    selected = np.flatnonzero(family_rows[0][rows])
    witness = np.full((len(days),7),np.nan)
    if len(selected):
        order = selected[np.lexsort((table[selected,1],table[selected,2],table[selected,4],row_days[selected]))]
        _, first = np.unique(row_days[order],return_index=True)
        chosen = order[first]
        witness[inverse[chosen]] = np.column_stack((peri[chosen],table[chosen,0],table[chosen,1],
                                                    table[chosen,2],table[chosen,3],table[chosen,4],
                                                    np.full(len(chosen),int(spkid))))
    pairs = np.unique(row_days*len(stats)+rows)
    return index, spkid, dict(days=days,minima=minima,winners=rows[best],pairs=pairs,witness=witness,
                             peri_min=float(peri.min()),peri_max=float(peri.max()),
                             soi_min=float((table[:,0] if reference=="SOI" else table[:,0]+stats[rows,7]).min()),
                             soi_max=float((table[:,0] if reference=="SOI" else table[:,0]+stats[rows,7]).max()))


def main():
    output = plots.OUTPUT_DIR/"copertura_temporale_famiglie"
    output.mkdir(parents=True,exist_ok=True)
    files = plots.discover_transfer_files(plots.TRANSFER_DIR)
    infos = [plots.read_transfer_header(path) for path in files]
    catalogs = {info["catalog"] for info in infos}
    if len(catalogs)!=1:
        raise ValueError("I file appartengono a cataloghi diversi.")
    catalog = catalogs.pop()
    stats = plots.load_table(catalog,14)
    stats = stats[np.argsort(stats[:,0])]
    members = plots.load_table(catalog.with_name("trajectory_exit_catalog.txt"),12)
    eps,ecc = members[:,9],members[:,10]
    valid = (members[:,1]>=0)&(eps<0)&(ecc>=0)&(ecc<1)
    rp = np.full(len(members),np.nan)
    np.divide(-plots.MU_EARTH*(1-ecc),2*eps,out=rp,where=eps!=0)
    valid &= rp>plots.R_EARTH_KM
    eligible_ids = np.unique(members[valid,1]).astype(int)
    eligible = np.isin(stats[:,0],eligible_ids)
    family_ids = []
    for tag in ("P97","P99_5"):
        with np.load(plots.OUTPUT_DIR/f"traiettorie_comuni_sei_mappe_{tag}"/"selezione.npz") as data:
            family_ids.append(np.unique(data["members"][:,1]).astype(int))
    with np.load(plots.OUTPUT_DIR/"intersezione_orbite_parcheggio"/"audit_baseline.npz") as data:
        family_ids.append(np.unique(data["members"][:,1]).astype(int))
    del members,eps,ecc,rp,valid
    global _CONFIG
    _CONFIG = (stats,eligible,[np.isin(stats[:,0],ids) for ids in family_ids])
    tasks = [(i,path,info["epoch_reference"],info["spkid"]) for i,(path,info) in enumerate(zip(files,infos))]
    results = []
    started = last_report = time.monotonic()
    print(f"Analisi di tutte le opportunità di {len(files)} asteroidi; epoca al perigeo del modello medio.",flush=True)
    with get_context("fork").Pool(8) as pool:
        for result in pool.imap_unordered(analyze_file,tasks,chunksize=4):
            results.append(result)
            if time.monotonic()-last_report>20:
                print(f"Letti {len(results)}/{len(files)} asteroidi; {time.monotonic()-started:.0f} s",flush=True)
                last_report=time.monotonic()
    nonempty = [r[2] for r in results if r[2] is not None]
    first = min(int(r["days"].min()) for r in nonempty)
    last = max(int(r["days"].max()) for r in nonempty)
    days = np.arange(first,last+1)
    minimum = np.full((len(FAMILY_NAMES),len(files),len(days)),np.inf)
    availability = np.zeros((len(stats),len(days)),dtype=np.int32)
    optimal = np.zeros_like(availability)
    witnesses = np.full((len(days),7),np.nan)
    for index,spkid,data in results:
        if data is None: continue
        columns=data["days"]-first
        minimum[:,index,columns]=data["minima"]
        np.add.at(optimal,(data["winners"],columns),1)
        pairs=data["pairs"]
        np.add.at(availability,(pairs%len(stats),pairs//len(stats)-first),1)
        candidate=data["witness"]
        better=np.isfinite(candidate[:,5])&(~np.isfinite(witnesses[columns,5])|(candidate[:,5]<witnesses[columns,5]))
        witnesses[columns[better]]=candidate[better]
    # Supplement the geometric P97 family to cover all dates represented by
    # the catalog. Greedy coverage is not a proof of a minimum-size family.
    observed=availability.any(axis=0)
    extended=np.isin(stats[:,0],family_ids[0])
    covered=availability[extended].any(axis=0)
    additions=[]
    while np.any(observed&~covered):
        missing=observed&~covered
        gains=(availability[:,missing]>0).sum(axis=1)
        options=np.flatnonzero(gains==gains.max())
        chosen=options[np.argmax(availability[options][:,missing].sum(axis=1))]
        if gains[chosen]==0:
            raise ValueError("Impossibile coprire un giorno che risulta disponibile nel catalogo.")
        additions.append(int(stats[chosen,0]))
        extended[chosen]=True
        covered |= availability[chosen]>0
    print(f"Famiglia ampliata: {len(family_ids[0])} + {len(additions)} cluster; verifica di tutte le opportunità.",flush=True)
    _CONFIG=(stats,eligible,[extended,*_CONFIG[2][1:]])
    extra_minimum=np.full((len(files),len(days)),np.inf)
    extra_witnesses=np.full((len(days),7),np.nan)
    with get_context("fork").Pool(8) as pool:
        for index,spkid,data in pool.imap_unordered(analyze_file,tasks,chunksize=4):
            if data is None: continue
            columns=data["days"]-first
            extra_minimum[index,columns]=data["minima"][1]
            candidate=data["witness"]
            better=np.isfinite(candidate[:,5])&(~np.isfinite(extra_witnesses[columns,5])|
                                               (candidate[:,5]<extra_witnesses[columns,5]))
            extra_witnesses[columns[better]]=candidate[better]
    minimum=np.concatenate((minimum,extra_minimum[None]),axis=0)
    family_names=(*FAMILY_NAMES,"Famiglia P97 ampliata nel tempo")
    available=np.isfinite(minimum)
    counts=available.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        medians=np.nanmedian(np.where(available,minimum,np.nan),axis=1)
        best=np.min(minimum,axis=1)
    best[~np.isfinite(best)]=np.nan
    # Compare on the dates where any catalog family supplies an opportunity.
    observed=counts[0]>0
    family_summaries=[]
    for i,name in enumerate(family_names):
        present=counts[i]>0
        family_summaries.append(dict(name=name,days_with_opportunities=int((present&observed).sum()),
                                     observed_days=int(observed.sum()),coverage_fraction=float(present[observed].mean()),
                                     minimum_available_asteroids=int(counts[i,observed].min()),
                                     median_available_asteroids=float(np.median(counts[i,observed])),
                                     gap_jd=(days[observed&~present]-.5).tolist()))
        print(name,f": {int((present&observed).sum())}/{observed.sum()} giorni; minimo {counts[i,observed].min()} asteroidi",flush=True)
    np.savez_compressed(output/"copertura_giornaliera.npz",jd_day_start=days-.5,minimum_dv=minimum,
                        reachable_asteroids=counts,median_dv=medians,best_dv=best,
                        cluster_ids=stats[:,0].astype(int),cluster_availability=availability,
                        cluster_optimal_asteroids=optimal,witness_P97=witnesses,witness_extended=extra_witnesses,
                        asteroid_spkids=np.array([int(info["spkid"]) for info in infos]))
    with (output/"partenze_famiglia_P97.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["jd_day_start_tdb","jd_perigee_tdb","jd_reference_tdb","cluster_id","jd_arrival_tdb",
                         "tof_from_soi_days","dv_rendezvous_km_s","asteroid_spkid"])
        for day,witness in zip(days,witnesses):
            if np.isfinite(witness[5]): writer.writerow([day-.5,*witness])
    with (output/"partenze_famiglia_ampliata.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["jd_day_start_tdb","jd_perigee_tdb","jd_reference_tdb","cluster_id","jd_arrival_tdb",
                         "tof_from_soi_days","dv_rendezvous_km_s","asteroid_spkid"])
        for day,witness in zip(days,extra_witnesses):
            if np.isfinite(witness[5]): writer.writerow([day-.5,*witness])
    summary=dict(reference="Giorno del perigeo del modello medio, JD TDB, intervalli [inizio, inizio+1)",
                 caveat="Presenza di almeno un campione nel giorno, non disponibilità continua a ogni istante",
                 family_interpretation="Tutte le opportunità del modello medio dei cluster contenenti i membri selezionati",
                 maximum_dv=plots.MAX_DV_KM_S,maximum_duration_days=plots.MAX_DURATION_DAYS,
                 peri_min_jd=min(r["peri_min"] for r in nonempty),peri_max_jd=max(r["peri_max"] for r in nonempty),
                 soi_min_jd=min(r["soi_min"] for r in nonempty),soi_max_jd=max(r["soi_max"] for r in nonempty),
                 family_cluster_ids={FAMILY_NAMES[i+1]:ids.tolist() for i,ids in enumerate(family_ids)},
                 families=family_summaries,extended_family_cluster_ids=stats[extended,0].astype(int).tolist(),
                 added_cluster_ids=additions,extension_method="Greedy per copertura dei giorni; non minimo numero garantito di cluster",
                 no_catalog_opportunity_jd=(days[~observed]-.5).tolist())
    (output/"riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
    x=days-.5-2451545.+mdates.date2num(np.datetime64("2000-01-01T12:00:00"))
    colors=("#222222","#0072b2","#d55e00","#009e73","#cc0077")
    with PdfPages(output/"copertura_temporale_famiglie.pdf") as pdf:
        fig,axes=plt.subplots(3,1,figsize=(16,11),sharex=True,layout="constrained")
        for i,(name,color) in enumerate(zip(family_names,colors)):
            axes[0].plot(x,counts[i],label=name,color=color,lw=1)
            axes[1].plot(x,medians[i],color=color,lw=1)
            axes[2].plot(x,best[i],color=color,lw=1)
        axes[0].set_ylabel("Asteroidi con opportunità nel giorno")
        axes[1].set_ylabel("Mediana dei minimi ΔV [km/s]")
        axes[2].set_ylabel("Minimo ΔV a un bersaglio [km/s]")
        axes[0].legend(ncol=2,fontsize=9)
        for ax in axes:
            ax.grid(alpha=.2)
        locator=mdates.AutoDateLocator()
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[-1].set_xlabel("Data del perigeo del modello medio — TDB")
        fig.suptitle("Disponibilità temporale delle famiglie — tutte le opportunità salvate",fontsize=16)
        fig.supxlabel("Contenitori di un giorno; presenza di campioni, non copertura continua. Minimo ΔV su arrivi e cluster della famiglia, per asteroide.\n"
                      "Mediana sui bersagli disponibili in ciascuna famiglia; solo impulso di rendez-vous, prestazioni del modello medio del cluster.",fontsize=9)
        fig.savefig(output/"01_disponibilita_e_dv.png",dpi=180); pdf.savefig(fig); plt.close(fig)
        rows=np.flatnonzero(extended)
        order=np.argsort(availability[rows].sum(axis=1))[::-1]; rows=rows[order]
        fig,ax=plt.subplots(figsize=(16,11),layout="constrained")
        image=ax.imshow(availability[rows],aspect="auto",origin="upper",extent=(x[0]-.5,x[-1]+.5,len(rows)-.5,-.5),
                        cmap="viridis",interpolation="none")
        ax.set_yticks(np.arange(len(rows)),stats[rows,0].astype(int),fontsize=7)
        ax.set_ylabel("Cluster della famiglia ampliata (* = aggiunto per copertura temporale)")
        ax.set_xlabel("Data del perigeo del modello medio — TDB")
        locator=mdates.AutoDateLocator(); ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_yticklabels([str(int(stats[row,0]))+(" *" if int(stats[row,0]) in additions else "") for row in rows],fontsize=7)
        ax.set_title("Quali cluster della famiglia ampliata sono disponibili a ogni data?",fontsize=16)
        fig.colorbar(image,ax=ax,label="Asteroidi distinti con almeno un trasferimento nel giorno")
        fig.supxlabel("Tutte le opportunità ammesse dai filtri correnti; colonne giornaliere e geometrie orientate con la fase lunare all’epoca ETD.",fontsize=9)
        fig.savefig(output/"02_calendario_cluster_P97.png",dpi=180); pdf.savefig(fig); plt.close(fig)
    print(f"Analisi temporale salvata in {output}",flush=True)


if __name__=="__main__":
    main()
