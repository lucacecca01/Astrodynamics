"""Ricolora le 12 fasce a +/-10000 km con il solo DV circolare(rp) -> escape."""

import csv
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from GA_18_parking_circle_metrics import draw
from GA_18_perigee_parking_circles import CURVE_DATA, OUTPUT, SELECTION


ROOT = OUTPUT / "estensione_radiale"
DESTINATION = ROOT / "delta_v_immissione"
MU_EARTH = 6.67430e-20 * 5.974e24
TOLERANCE_KM = 10000


def prepare():
    with np.load(ROOT / "asteroidi_vsoi/perigei_asteroidi_vsoi.npz") as source:
        data = {key: source[key] for key in ("source_ids", "rp_km", "synodic_xy_km")}
    with np.load(SELECTION) as source:
        ids, first, inverse = np.unique(source["source_ids"], return_index=True, return_inverse=True)
        assert np.array_equal(ids, data["source_ids"])
        assert source["event_is_periapsis"].all()
        injection = source["delta_v_injection_km_s"][first]
        assert np.array_equal(source["delta_v_injection_km_s"], injection[inverse])
        assert np.array_equal(source["rp_km"][first], data["rp_km"])
        assert np.allclose(source["delta_v_total_km_s"],
                           source["delta_v_hohmann_km_s"] + source["delta_v_injection_km_s"],
                           rtol=0, atol=1e-12)
    # Controllo indipendente dei costi salvati tramite gli elementi osculatori.
    with np.load(CURVE_DATA / "selezione_istanze.npz") as source:
        members = source["members"]
    rows = np.searchsorted(members[:, 0], ids)
    assert np.array_equal(members[rows, 0], ids)
    eps, ecc = members[rows, 9], members[rows, 10]
    rp = -MU_EARTH * (1-ecc)/(2*eps)
    expected = np.sqrt(MU_EARTH/rp) * ecc/(np.sqrt(1+ecc)+1)
    assert np.allclose(rp, data["rp_km"], rtol=1e-12, atol=1e-6)
    assert np.allclose(injection, expected, rtol=1e-12, atol=1e-12)
    data["delta_v_injection_km_s"] = injection
    original = json.loads((ROOT / "delta_v_totale/conteggi_estesi.json").read_text())
    bands = original["bands"][str(TOLERANCE_KM)]
    assignments = np.full(len(ids), -1, dtype=int)
    for number, band in enumerate(bands):
        indices = np.searchsorted(ids, band["source_ids"])
        assert np.array_equal(ids[indices], band["source_ids"])
        assert np.array_equal(np.flatnonzero(np.abs(data["rp_km"]-band["radius_km"]) <= TOLERANCE_KM), indices)
        assert len(indices) == band["count"] and np.all(assignments[indices] < 0)
        assignments[indices] = number
        values = injection[indices]
        band.update(delta_v_injection_mean_km_s=float(values.mean()),
                    delta_v_injection_min_km_s=float(values.min()),
                    delta_v_injection_max_km_s=float(values.max()))
    assert len(bands) == 12 and len(ids) == 341 and np.all(assignments >= 0)
    data["band_indices"] = assignments
    report = dict(selection=str(SELECTION), distinct_perigees=len(ids), radial_bands=len(bands),
                  tolerance_km=TOLERANCE_KM, bands=bands,
                  delta_v_definition="sqrt(mu_E/rp) * (sqrt(1+e)-1); circolare(rp) -> escape, stesso piano e verso.",
                  point_color="Delta V di immissione individuale",
                  circle_color="Media dei Delta V individuali nella fascia",
                  color_limits_km_s=[float(injection.min()), float(injection.max())],
                  excluded_cost="Trasferimento Hohmann dalla LEO al parcheggio",
                  l1_geocentric_xy_km=original["l1_geocentric_xy_km"])
    return data, report


def main():
    data, report = prepare()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metric = dict(key="delta_v_immissione", field="delta_v_injection_mean_km_s",
                  point="delta_v_injection_km_s", column="ΔV [km/s]", cmap="viridis_r",
                  title="Immissione dalla circolare all'escape",
                  label="ΔV dalla circolare(rₚ) all'escape [km/s]",
                  note="Pallini: ΔV individuale. Cerchi e tabella: ΔV medio dei perigei nella fascia.\n"
                       "Solo immissione dalla circolare(rₚ) nell'escape; stesso piano e verso. Costo Hohmann escluso.")
    fig = draw(metric, TOLERANCE_KM, report["bands"], data, Normalize(*report["color_limits_km_s"]))
    fig.suptitle(f"Immissione dalla circolare all'escape — tolleranza ±{TOLERANCE_KM:,} km\n"
                 f"341 perigei distinti · raggio massimo {data['rp_km'].max():,.0f} km", fontsize=17)
    fig.texts[-1].set_text(metric["note"])
    stem = DESTINATION / f"circolari_tolleranza_{TOLERANCE_KM}km"
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    np.savez_compressed(DESTINATION / "perigei_delta_v_immissione.npz", **data)
    (DESTINATION / "riepilogo.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ("radius_km", "tolerance_km", "count", "delta_v_injection_mean_km_s",
              "delta_v_injection_min_km_s", "delta_v_injection_max_km_s")
    with (DESTINATION / "statistiche_fasce.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["bands"])
    print(json.dumps(dict(output=str(stem.with_suffix(".png")), perigees=len(data["source_ids"]),
                         bands=len(report["bands"]), injection_range_km_s=report["color_limits_km_s"]), indent=2))


if __name__ == "__main__":
    main()
