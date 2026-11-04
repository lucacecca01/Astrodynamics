"""Ricolora le stesse circolari con asteroidi distinti e velocita alla SOI.

Le associazioni agli asteroidi vengono dai migliori trasferimenti GA18 del
modello medio del cluster. Non esegue nuovi Lambert o raccordi individuali.
"""

import csv
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MaxNLocator

from GA_18_perigee_parking_circles import (
    BASE, CURVE_DATA, D_MOON_KM, EXTENDED_TOLERANCES_KM, OUTPUT, R_EARTH_KM, SELECTION,
    covering_bands, draw_extended, draw_l1, l1_position_km,
)


DESTINATION = OUTPUT / "estensione_radiale/asteroidi_vsoi"
METRICS = (
    dict(key="asteroidi_raggiunti", field="asteroids_distinct", point="asteroids_per_source",
         title="Asteroidi distinti raggiunti", column="Asteroidi", cmap="viridis",
         label="Numero di asteroidi distinti",
         note="Pallini: asteroidi del singolo membro. Cerchi: unione degli asteroidi dei membri nella fascia."),
    dict(key="vsoi_massima", field="v_soi_max_km_s", point="v_soi_km_s",
         title="Velocita SOI massima", column="Max V [km/s]", cmap="inferno",
         label="Velocita geocentrica inerziale alla SOI [km/s]",
         note="Pallini: velocita del singolo membro. Cerchi: massimo dei moduli delle velocita nella fascia."),
    dict(key="vsoi_media", field="v_soi_mean_km_s", point="v_soi_km_s",
         title="Velocita SOI media", column="Media V [km/s]", cmap="inferno",
         label="Velocita geocentrica inerziale alla SOI [km/s]",
         note="Pallini: velocita del singolo membro. Cerchi: media dei moduli delle velocita nella fascia."),
)


def prepare():
    root = OUTPUT / "estensione_radiale"
    with np.load(root / "delta_v_totale/perigei_delta_v_totale.npz") as data:
        ids, radii, points, dv = (data[k] for k in
            ("source_ids", "rp_km", "synodic_xy_km", "delta_v_total_km_s"))
    assert np.array_equal(ids, np.unique(ids))
    with np.load(SELECTION) as data:
        sources = data["source_ids"]
        winner_indices = data["winner_indices"]
        cluster_ids = data["cluster_ids"]
        phases = data["phase_etd_deg"]
    winners = json.loads((CURVE_DATA / "riepilogo.json").read_text())["winners"]
    assert np.all(winner_indices >= 0) and winner_indices.max() < len(winners)
    assert np.array_equal(cluster_ids, np.array([w["cluster_id"] for w in winners])[winner_indices])
    assert np.allclose(phases, np.array([w["phase_etd_deg"] for w in winners])[winner_indices],
                       rtol=0, atol=1e-10)
    asteroid_keys, asteroid_indices = np.unique(
        np.array([str(w["spkid"]) for w in winners])[winner_indices], return_inverse=True)
    source_indices = np.searchsorted(ids, sources)
    assert np.array_equal(ids[source_indices], sources)
    # Una coppia membro-asteroide vale uno anche se presente a piu ETD.
    associations = np.zeros((len(ids), len(asteroid_keys)), dtype=bool)
    associations[source_indices, asteroid_indices] = True
    with np.load(CURVE_DATA / "selezione_istanze.npz") as data:
        members = data["members"]
    rows = np.searchsorted(members[:, 0], ids)
    assert np.array_equal(members[rows, 0], ids)
    assert np.array_equal(members[rows[source_indices], 1], cluster_ids)
    speed = np.linalg.norm(members[rows, 2:5], axis=1)
    assert np.isfinite(speed).all() and (speed > 0).all()
    point_data = dict(source_ids=ids, rp_km=radii, synodic_xy_km=points,
                      delta_v_total_km_s=dv, v_soi_km_s=speed,
                      asteroids_per_source=associations.sum(axis=1),
                      asteroid_spkids=asteroid_keys, asteroid_associations=associations)
    report = json.loads((root / "conteggi_estesi.json").read_text())
    # Ridisegna le fasce per includere tutti i punti, senza duplicare i membri.
    report["bands"] = {
        str(tolerance): covering_bands(radii, ids, tolerance)
        for tolerance in EXTENDED_TOLERANCES_KM
    }
    for bands in report["bands"].values():
        covered = np.zeros(len(ids), dtype=bool)
        for band in bands:
            indices = np.searchsorted(ids, band["source_ids"])
            assert np.array_equal(ids[indices], band["source_ids"])
            mask = np.abs(radii-band["radius_km"]) <= band["tolerance_km"]
            assert np.array_equal(np.flatnonzero(mask), indices)
            assert len(indices) == band["count"] and not covered[indices].any()
            covered[indices] = True
            band.update(
                asteroids_distinct=int(associations[indices].any(axis=0).sum()),
                v_soi_max_km_s=float(speed[indices].max()),
                v_soi_mean_km_s=float(speed[indices].mean()),
            )
        assert covered.all(), "Copertura incompleta dei perigei."
    count_max = max(b["asteroids_distinct"] for bands in report["bands"].values() for b in bands)
    # La stessa scala confronta le tolleranze; media e massimo V condividono i limiti.
    count_norm = Normalize(0, max(count_max, int(point_data["asteroids_per_source"].max())))
    speed_norm = Normalize(float(speed.min()), float(speed.max()))
    report.update(
        rule="Copertura completa ordinata: fascia [primo raggio non coperto, raggio+2*tolleranza].",
        note="Minimo numero di intervalli radiali di larghezza 2*tolleranza; ogni source_id conta una volta.",
        coverage_per_tolerance={tol: sum(band["count"] for band in bands) for tol, bands in report["bands"].items()},
        asteroid_definition="Unione di SPK-ID distinti associati ai membri, senza sommare ETD o traiettorie ripetute.",
        association_model="Miglior trasferimento di ogni asteroide, modello medio del cluster GA18.",
        speed_definition="Norma della velocita SOI geocentrica inerziale, membri[:, 2:5], un peso per source_id.",
        color_limits_asteroids=[count_norm.vmin, count_norm.vmax],
        color_limits_speed_km_s=[speed_norm.vmin, speed_norm.vmax],
        distinct_asteroids=len(asteroid_keys),
        l1_geocentric_xy_km=[l1_position_km(), 0.0],
    )
    return point_data, report, count_norm, speed_norm


def draw(metric, tolerance, bands, data, norm):
    colors = plt.get_cmap(metric["cmap"])(norm([b[metric["field"]] for b in bands]))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([.055, .21, .48, .66])
    table_ax = fig.add_axes([.58, .17, .39, .70])
    points = data["synodic_xy_km"]
    fig.suptitle(f"{metric['title']} — tolleranza ±{int(tolerance):,} km\n"
                 f"{len(points)} perigei distinti · stesse circolari fino a {data['rp_km'].max():,.0f} km", fontsize=17)
    for band, color in zip(bands, colors):
        radius, tol = band["radius_km"], band["tolerance_km"]
        ax.add_patch(Wedge((0, 0), radius+tol, 0, 360, width=2*tol,
                           facecolor=color, edgecolor="none", alpha=.06))
        ax.add_patch(Circle((0, 0), radius, fill=False, color=color, linewidth=.8))
    scatter = ax.scatter(*points.T, c=data[metric["point"]], cmap=metric["cmap"], norm=norm,
                         s=24, edgecolor="0.15", linewidth=.3, zorder=4)
    cax = fig.add_axes([.105, .115, .38, .018])
    bar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    bar.set_label(metric["label"], fontsize=10)
    if metric["key"] == "asteroidi_raggiunti":
        bar.locator = MaxNLocator(nbins=6, integer=True)
        bar.update_ticks()
    ax.add_patch(Circle((0, 0), R_EARTH_KM, color="royalblue", zorder=5))
    ax.annotate("Terra", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.plot(D_MOON_KM, 0, "o", color="0.3", markersize=4)
    ax.annotate("Luna", (D_MOON_KM, 0), xytext=(-5, 8),
                textcoords="offset points", ha="right", fontsize=9)
    draw_l1(ax)
    ax.set(xlabel="x geocentrica sinodica [km]", ylabel="y geocentrica sinodica [km]",
           xlim=(-410000, 410000), ylim=(-410000, 410000),
           title=f"{len(bands)} fasce disgiunte · {sum(b['count'] for b in bands)}/{len(points)} perigei inclusi")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    table_ax.axis("off")
    table_ax.set_title("Raggi dal centro terrestre, ordinati dall'interno all'esterno", fontsize=11)
    chunks = np.array_split(np.arange(len(bands)), 2 if len(bands) > 30 else 1)
    for column, indices in enumerate(chunks):
        rows = []
        for index in indices:
            value = bands[index][metric["field"]]
            label = str(value) if metric["key"] == "asteroidi_raggiunti" else f"{value:.3f}"
            rows.append([str(index+1), f"{bands[index]['radius_km']:,.0f}", str(bands[index]["count"]), label])
        table = table_ax.table(cellText=rows,
            colLabels=["#", "Raggio [km]", "Perigei", metric["column"]],
            cellLoc="center", colWidths=[.09, .40, .17, .34],
            bbox=[column/len(chunks)+.01, .04, 1/len(chunks)-.03, .9])
        table.auto_set_font_size(False)
        table.set_fontsize(8 if len(chunks) == 2 else 12)
        for row, index in enumerate(indices, 1):
            for col in range(4):
                table[row, col].set_facecolor((*colors[index][:3], .16))
                table[row, col].set_edgecolor("0.9")
    fig.text(.5, .035,
             metric["note"]+"\n"
             "Asteroidi: associazioni del modello medio GA18. Velocita: moduli individuali geocentrici; ogni membro conta una volta.",
             ha="center", va="center", fontsize=10.5)
    return fig


def main():
    data, report, count_norm, speed_norm = prepare()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DESTINATION / "perigei_asteroidi_vsoi.npz", **data)
    (DESTINATION / "riepilogo.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ["tolerance_km", "radius_km", "altitude_km", "count",
              "asteroids_distinct", "v_soi_max_km_s", "v_soi_mean_km_s"]
    with (DESTINATION / "statistiche_fasce.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for bands in report["bands"].values():
            writer.writerows(bands)
    with PdfPages(DESTINATION / "circolari_asteroidi_vsoi_tutte.pdf") as all_pdf:
        for metric in METRICS:
            output = DESTINATION / metric["key"]
            output.mkdir(exist_ok=True)
            norm = count_norm if metric["key"] == "asteroidi_raggiunti" else speed_norm
            with (PdfPages(output / "circolari_tutte_tolleranze.pdf") as metric_pdf,
                  PdfPages(output / "circolari_due_tolleranze.pdf") as first_two_pdf):
                for tolerance, bands in report["bands"].items():
                    fig = draw(metric, tolerance, bands, data, norm)
                    fig.savefig(output / f"circolari_tolleranza_{tolerance}km.png", dpi=180)
                    metric_pdf.savefig(fig)
                    if tolerance in ("1000", "5000"):
                        first_two_pdf.savefig(fig)
                    all_pdf.savefig(fig)
                    plt.close(fig)
    # Anche le tavole DV ricevono tutte le tolleranze e il medesimo punto L1.
    draw_extended(data["rp_km"], data["source_ids"], data["synodic_xy_km"])
    draw_extended(data["rp_km"], data["source_ids"], data["synodic_xy_km"],
                  total_dv=data["delta_v_total_km_s"])
    print(json.dumps(dict(output=str(DESTINATION), figures=len(METRICS)*len(report["bands"]),
                         l1_radius_km=l1_position_km(),
                         asteroids=report["distinct_asteroids"],
                         speed_range_km_s=report["color_limits_speed_km_s"]), indent=2))


if __name__ == "__main__":
    main()
