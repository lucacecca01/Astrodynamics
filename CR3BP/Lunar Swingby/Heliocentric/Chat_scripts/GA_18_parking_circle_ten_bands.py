"""Tutti i confronti classici con dieci fasce occupate e copertura completa.

Adatta la semiampiezza comune al minimo chilometro intero che permette dieci
fasce, riusando i render originali senza profili o palette di contrasto.
"""

import csv
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

from GA_18_perigee_parking_circles import CURVE_DATA, OUTPUT, covering_bands, l1_position_km
from GA_18_parking_circle_metrics import METRICS as CATALOG_METRICS, draw as draw_catalog
from GA_18_parking_circle_angular_coverage import angular_coverage, angular_weights, draw as draw_angular
from GA_18_parking_circle_waiting_times import (
    MU_EARTH, N_MOON_RAD_S, PARKING_DIRECTION, METRICS as WAIT_METRICS,
    waiting_statistics, draw as draw_waiting,
)


ROOT = OUTPUT / "estensione_radiale"
INPUT = ROOT / "asteroidi_vsoi"
DESTINATION = ROOT / "10_fasce"
N_BANDS = 10
ANGULAR_RESOLUTION_DEG = 5.0


def ten_occupied_bands(radii, source_ids):
    """Ricerca la minima semiampiezza intera; il dataset corrente da dieci fasce."""
    lower, upper = 1, int(np.ceil(np.ptp(radii)/2))
    while lower < upper:
        middle = (lower+upper)//2
        if len(covering_bands(radii, source_ids, middle)) <= N_BANDS:
            upper = middle
        else:
            lower = middle+1
    bands = covering_bands(radii, source_ids, lower)
    if len(bands) != N_BANDS:
        raise ValueError("Il dataset non consente esattamente dieci fasce con questo criterio.")
    assert all(band["count"] > 0 for band in bands)
    return bands, lower


def prepare():
    with np.load(INPUT / "perigei_asteroidi_vsoi.npz") as source:
        data = {key: source[key] for key in source.files}
    ids, radii = data["source_ids"], data["rp_km"]
    points = data["synodic_xy_km"]
    assert np.array_equal(ids, np.unique(ids))
    assert np.allclose(np.linalg.norm(points, axis=1), radii, rtol=0, atol=1e-5)
    with np.load(CURVE_DATA / "selezione_istanze.npz") as source:
        members = source["members"]
    rows = np.searchsorted(members[:, 0], ids)
    assert np.array_equal(members[rows, 0], ids)
    features = np.load(CURVE_DATA / "reconstructed_features.npy", mmap_mode="r")
    assert np.all(features[rows, 3] == PARKING_DIRECTION)
    angles = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360)
    bands, halfwidth = ten_occupied_bands(radii, ids)
    assigned = np.full(len(ids), -1, dtype=int)
    per_band_fields = ("maximum_wait_hours", "mean_wait_hours", "angular_score_deg_perigees",
                       "angular_score_weighted_deg_perigees", "effective_perigee_count")
    for field in per_band_fields:
        data[field] = np.full(len(ids), np.nan)
    point_weights = np.full(len(ids), np.nan)
    for number, band in enumerate(bands):
        indices = np.searchsorted(ids, band["source_ids"])
        assert np.array_equal(ids[indices], band["source_ids"])
        assert np.all(assigned[indices] == -1)
        assigned[indices] = number
        band.update(
            asteroids_distinct=int(data["asteroid_associations"][indices].any(axis=0).sum()),
            v_soi_max_km_s=float(data["v_soi_km_s"][indices].max()),
            v_soi_mean_km_s=float(data["v_soi_km_s"][indices].mean()),
            delta_v_total_mean_km_s=float(data["delta_v_total_km_s"][indices].mean()),
            delta_v_total_min_km_s=float(data["delta_v_total_km_s"][indices].min()),
            delta_v_total_max_km_s=float(data["delta_v_total_km_s"][indices].max()),
        )
        band.update(angular_coverage(angles[indices]))
        point_weights[indices] = angular_weights(angles[indices], ANGULAR_RESOLUTION_DEG)
        effective = float(point_weights[indices].sum())
        band.update(effective_perigee_count=effective, angular_resolution_deg=ANGULAR_RESOLUTION_DEG,
                    crowding_factor=effective/band["count"],
                    angular_score_weighted_deg_perigees=effective*band["angle_span_deg"])
        parking_rate = PARKING_DIRECTION*np.sqrt(MU_EARTH/band["radius_km"]**3)
        relative_rate = np.degrees(parking_rate-N_MOON_RAD_S)*3600
        band.update(waiting_statistics(angles[indices], relative_rate))
        band["parking_angular_velocity_rad_s"] = float(parking_rate)
        for field in per_band_fields:
            data[field][indices] = band[field]
    assert np.all(assigned >= 0) and sum(band["count"] for band in bands) == len(ids)
    assert all(np.isfinite(data[field]).all() for field in per_band_fields)
    data.update(band_indices=assigned, angle_deg=angles, angular_weights=point_weights)
    previous = json.loads((INPUT / "riepilogo.json").read_text())
    previous_wait = json.loads((ROOT / "tempi_attesa/scala_dettaglio/riepilogo.json").read_text())
    norms = {
        "asteroidi_raggiunti": Normalize(0, max(previous["color_limits_asteroids"][1],
                                                  max(b["asteroids_distinct"] for b in bands))),
        "vsoi_massima": Normalize(*previous["color_limits_speed_km_s"]),
        "vsoi_media": Normalize(*previous["color_limits_speed_km_s"]),
        "delta_v_totale": Normalize(float(data["delta_v_total_km_s"].min()), float(data["delta_v_total_km_s"].max())),
        "copertura_angolare": Normalize(0, max(b["angular_score_deg_perigees"] for b in bands)),
        "copertura_angolare_pesata": Normalize(0, max(b["angular_score_weighted_deg_perigees"] for b in bands)),
    }
    for metric in WAIT_METRICS:
        lower = float(data[metric["field"]].min())
        upper = previous_wait["color_scales"][metric["key"]]["upper_hours"]
        assert upper > lower
        norms[metric["key"]] = Normalize(lower, upper, clip=False)
    report = dict(
        source=str(INPUT), radial_bands=N_BANDS, occupied_bands=len(bands),
        distinct_perigees=len(ids), perigees_assigned_once=int(np.count_nonzero(assigned >= 0)),
        halfwidth_km=halfwidth, full_width_km=2*halfwidth,
        rule="Dieci fasce occupate; minima semiampiezza comune al chilometro intero, copertura ordinata completa.",
        bands_at_one_km_less=len(covering_bands(radii, ids, halfwidth-1)),
        frame="Geocentrico sinodico Terra-Luna", l1_geocentric_xy_km=[l1_position_km(), 0.0],
        parking_direction=PARKING_DIRECTION, angular_resolution_deg=ANGULAR_RESOLUTION_DEG,
        asteroid_definition="Unione di SPK-ID distinti della fascia, nel modello medio GA18.",
        velocity_definition="Moduli delle velocita SOI geocentriche individuali; un peso per membro.",
        delta_v_definition="Media dei costi individuali: Hohmann LEO200 -> circolare(rp) + immissione nell'escape.",
        waiting_definition="Circolare retrograda al centro della fascia; omega_rel=-sqrt(mu_E/R^3)-n_L; media su fase uniforme.",
        waiting_scale="Stile precedente: scala lineare, cap superiori P75 precedenti, nuovi minimi reali.",
        color_limits={key: [float(norm.vmin), float(norm.vmax)] for key, norm in norms.items()},
        model_note="Raggruppamento geometrico; nessun nuovo raccordo dinamico o Lambert.",
        bands=bands,
    )
    return data, report, norms


def finish(fig, title, report, footer):
    fig.suptitle(f"{title} — 10 fasce radiali\n"
                 f"341 perigei distinti · semiampiezza comune ±{report['halfwidth_km']:,} km", fontsize=17)
    fig.texts[-1].set_text(footer)
    return fig


def main():
    data, report, norms = prepare()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DESTINATION / "perigei_10_fasce.npz", **data)
    (DESTINATION / "riepilogo.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ["radius_km", "tolerance_km", "count", "asteroids_distinct", "v_soi_max_km_s", "v_soi_mean_km_s",
              "delta_v_total_mean_km_s", "largest_gap_deg", "angle_span_deg", "effective_perigee_count",
              "angular_score_deg_perigees", "angular_score_weighted_deg_perigees", "maximum_wait_hours", "mean_wait_hours"]
    with (DESTINATION / "statistiche_10_fasce.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["bands"])
    bands, halfwidth = report["bands"], report["halfwidth_km"]
    produced = []

    def save(fig, filename, pdf):
        fig.savefig(DESTINATION / f"{filename}.png", dpi=180)
        fig.savefig(DESTINATION / f"{filename}.pdf")
        pdf.savefig(fig)
        produced.append(filename)
        plt.close(fig)

    with PdfPages(DESTINATION / "tutti_grafici_10_fasce.pdf") as pdf:
        for metric in CATALOG_METRICS:
            fig = draw_catalog(metric, halfwidth, bands, data, norms[metric["key"]])
            finish(fig, metric["title"], report, metric["note"]+"\n"
                   "Dieci fasce occupate; ogni perigeo e assegnato una volta. Associazioni agli asteroidi dal modello medio GA18.")
            save(fig, metric["key"], pdf)
        metric = dict(key="delta_v_totale", field="delta_v_total_mean_km_s", point="delta_v_total_km_s",
                      title="Delta V totale di partenza", column="ΔV [km/s]", cmap="viridis_r",
                      label="ΔV totale di partenza da LEO 200 km [km/s]", note="")
        fig = draw_catalog(metric, halfwidth, bands, data, norms[metric["key"]])
        finish(fig, metric["title"], report,
               "Pallini: ΔV individuale. Cerchi e tabella: media dei costi dei membri nella fascia.\n"
               "Due impulsi Hohmann da LEO 200 km alla circolare(rp) + immissione nell'escape; stesso piano e verso.")
        save(fig, metric["key"], pdf)
        for metric in WAIT_METRICS:
            norm = norms[metric["key"]]
            fig = draw_waiting(metric, halfwidth, bands, data, data[metric["field"]], norm, focused=True)
            finish(fig, metric["title"]+" — parcheggio retrogrado", report,
                   f"Scala lineare fino a {norm.vmax:.1f} h; tempi superiori saturati, valori reali in tabella. Pallini e cerchio: statistica della fascia.\n"
                   "Circolare kepleriana retrograda e Luna circolare; media su fase iniziale uniforme. Finestre degli asteroidi separate.")
            save(fig, metric["key"], pdf)
        for weighted in (False, True):
            key = "copertura_angolare_pesata" if weighted else "copertura_angolare"
            field = "angular_score_weighted_deg_perigees" if weighted else "angular_score_deg_perigees"
            fig = draw_angular(halfwidth, bands, data, data[field], norms[key],
                               ANGULAR_RESOLUTION_DEG if weighted else None)
            note = ("S = span × N_eff; riduzione della ridondanza con distanza di riferimento 5 gradi."
                    if weighted else "S = span × N; span = 360 gradi meno il massimo intervallo angolare vuoto.")
            finish(fig, "Copertura angolare pesata" if weighted else "Copertura angolare", report,
                   note+"\nArco colorato: span minimo contenente i perigei; tratteggio grigio: massimo gap vuoto. Tutti i 341 perigei inclusi.")
            save(fig, key, pdf)
    report["figures"] = produced
    (DESTINATION / "riepilogo.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(output=str(DESTINATION), figures=len(produced), occupied_bands=len(bands),
                         perigees=report["perigees_assigned_once"], halfwidth_km=halfwidth,
                         counts=[band["count"] for band in bands]), indent=2))


if __name__ == "__main__":
    main()
