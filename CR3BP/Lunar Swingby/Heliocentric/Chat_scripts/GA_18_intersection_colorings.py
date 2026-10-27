"""Le stesse 137 traiettorie P97/>100, colorate per V_SOI e tempo del perigeo."""

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

import GA_18_plots as plots
from GA_18_combined_trajectories import draw_combined, inertial_positions, trajectory_times


def main():
    root = plots.OUTPUT_DIR/"intersezione_orbite_parcheggio"
    with np.load(root/"P97_selezione.npz") as data:
        members, ids, scores = data["members"], data["source_ids"], data["scores"]
    with np.load(root/"audit_baseline.npz") as data:
        baseline = data["members"]
    if len(ids)!=137 or not np.all(scores>100):
        raise ValueError("La selezione non corrisponde alle 137 traiettorie richieste.")
    parent = json.loads((plots.OUTPUT_DIR/"traiettorie_percentile_individuale_99_99"/"riepilogo.json").read_text())
    cache = Path(parent["dense_cache"])
    cached_ids = np.load(cache/"source_ids.npy",mmap_mode="r")
    positions = np.searchsorted(cached_ids,ids)
    base_positions = np.searchsorted(cached_ids,baseline[:,0].astype(np.int64))
    cached_members = np.load(cache/"members.npy",mmap_mode="r")
    done = np.load(cache/"done.npy",mmap_mode="r")
    if (not np.array_equal(cached_ids[positions],ids) or not done[positions].all()
            or not np.array_equal(cached_members[positions],members)
            or not np.array_equal(cached_members[base_positions],baseline)):
        raise ValueError("Cache incoerente con la selezione o i limiti del grafico originale.")
    all_curves = np.load(cache/"curves.npy",mmap_mode="r")
    curves, base_curves = all_curves[positions],all_curves[base_positions]
    base_inertial = inertial_positions(base_curves,trajectory_times(baseline[:,5:7],101))
    limits = np.array([np.minimum(base_curves.min(axis=(0,2)),base_inertial.min(axis=(0,2))),
                       np.maximum(base_curves.max(axis=(0,2)),base_inertial.max(axis=(0,2)))])
    v_soi = np.linalg.norm(members[:,2:5],axis=1)
    t_perigee = members[:,5]
    if not np.isfinite(v_soi).all() or not np.all(t_perigee<0):
        raise ValueError("Colori non finiti o tempo del perigeo non precedente all’ETD.")
    summary = json.loads((root/"riepilogo.json").read_text())
    rec = next(r for r in summary["levels"] if r["percentile"]==97)
    output = root/"P97_colori_Vsoi_tperigeo"
    output.mkdir(parents=True,exist_ok=True)
    variants = (
        ("01_V_soi", v_soi, r"$V_{\rm SOI}=|\mathbf{v}_{\rm SOI}|$ [km/s]",
         "Colore: modulo della velocità geocentrica inerziale alla SOI di ciascuna traiettoria, dal catalogo."),
        ("02_t_perigeo", t_perigee, r"$t_{\rm perigeo}-t_{\rm ETD}$ [giorni]",
         "Colore: tempo del perigeo rispetto a ETD = 0; valori negativi indicano giorni prima dell’ETD."),
    )
    title = ("Intersezione: >100 asteroidi vincenti per traiettoria e P97 in tutte le sei mappe\n"
             f"137 traiettorie · 5 cluster · {rec['distinct_optimal_asteroids']} asteroidi distinti · proiezione "+r"$\varepsilon$–$e$")
    with PdfPages(output/"confronto_Vsoi_tperigeo.pdf") as pdf:
        for name,values,label,note in variants:
            image_path = output/f"{name}.png"
            fig = draw_combined(curves,members,values,label,image_path,len(baseline),title=title,
                                normalization=Normalize(float(values.min()),float(values.max())),
                                fixed_alpha=0.4,line_width=1.0,global_limits=limits,raster_chunks=True,
                                perigee_markers=True,return_figure=True)
            try:
                if name=="02_t_perigeo":
                    fig.axes[-1].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g} d"))
                fig.supxlabel(fig._supxlabel.get_text()+"\n"+note,fontsize=9)
                fig.savefig(image_path,dpi=180)
                fig.savefig(output/f"{name}.pdf",dpi=180)
                pdf.savefig(fig,dpi=180)
            finally:
                plots.plt.close(fig)
            print(f"Salvato {name}: {values.min():.6f} … {values.max():.6f}",flush=True)
    np.savez_compressed(output/"colori_traiettorie.npz",source_ids=ids,members=members,
                        v_soi_km_s=v_soi,t_perigee_from_etd_days=t_perigee)
    (output/"riepilogo.json").write_text(json.dumps(dict(
        source_selection=str(root/"P97_selezione.npz"),trajectories=len(ids),clusters=5,
        alpha=0.4,line_width=1.0,samples_per_trajectory=201,
        v_soi_definition="Norma delle componenti 2:5 del membro: velocità geocentrica inerziale alla SOI, km/s",
        time_definition="Colonna 5 del membro: giorni con segno rispetto a ETD=0, non rispetto alla SOI",
        v_soi_range_km_s=[float(v_soi.min()),float(v_soi.max())],
        t_perigee_range_days=[float(t_perigee.min()),float(t_perigee.max())]),indent=2)+"\n")
    print(f"Salvate entrambe le colorazioni in {output}",flush=True)


if __name__=="__main__":
    main()
