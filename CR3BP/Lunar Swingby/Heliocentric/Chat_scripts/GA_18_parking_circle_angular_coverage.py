"""Colora le circolari con span angolare * numero effettivo di perigei.

Riusa la copertura completa dei 341 perigei e le tre tolleranze gia salvate.
Gli angoli provengono dalle posizioni geocentriche sinodiche, non da omega.
La risoluzione angolare predefinita e 5 gradi; --unweighted riproduce N*span.
"""

import argparse
import csv
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Circle, Wedge

from GA_18_perigee_parking_circles import (
    D_MOON_KM, OUTPUT, R_EARTH_KM, draw_l1, l1_position_km,
)


INPUT = OUTPUT / "estensione_radiale/asteroidi_vsoi"
DESTINATION = OUTPUT / "estensione_radiale/copertura_angolare"


def angular_coverage(angles_deg):
    """Arco minimo contenente i punti: complemento del maggiore gap circolare."""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Serve almeno un angolo finito, in un vettore monodimensionale.")
    count = len(angles)
    unique_angles = np.unique(np.mod(angles, 360.0))
    gaps = np.diff(np.append(unique_angles, unique_angles[0]+360.0))
    index = int(np.argmax(gaps))
    gap = float(gaps[index]) if len(unique_angles) > 1 else 360.0
    span = max(0.0, min(360.0, 360.0-gap))
    gap_start = float(unique_angles[index])
    gap_end = float(unique_angles[(index+1) % len(unique_angles)])
    return dict(
        largest_gap_deg=gap, angle_span_deg=span,
        largest_gap_start_deg=gap_start, largest_gap_end_deg=gap_end,
        covered_arc_start_deg=gap_end, covered_arc_end_unwrapped_deg=gap_end+span,
        angular_score_deg_perigees=float(count*span),
    )


def angular_weights(angles_deg, resolution_deg):
    """Perigei coincidenti si dividono peso 1; distanti >= risoluzione pesano 1."""
    angles = np.asarray(angles_deg, dtype=float)
    if (angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all()
            or not np.isfinite(resolution_deg) or not 0 < resolution_deg <= 180):
        raise ValueError("Servono angoli finiti e una risoluzione in (0, 180] gradi.")
    angles = np.mod(angles, 360.0)
    differences = np.abs(angles[:, None]-angles[None, :])
    distance = np.minimum(differences, 360.0-differences)
    proximity = np.maximum(0.0, 1.0-distance/resolution_deg)
    return 1.0/proximity.sum(axis=1)  # Include se stesso, con prossimita 1.


def prepare(resolution_deg=None):
    report = json.loads((INPUT / "riepilogo.json").read_text())
    with np.load(INPUT / "perigei_asteroidi_vsoi.npz") as data:
        ids, radii, points = (data[key] for key in ("source_ids", "rp_km", "synodic_xy_km"))
    assert np.array_equal(ids, np.unique(ids))
    assert np.allclose(np.linalg.norm(points, axis=1), radii, rtol=0, atol=1e-5)
    angles = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
    point_scores, assignments, point_weights, weighted_scores = [], [], [], []
    for bands in report["bands"].values():
        scores = np.full(len(ids), np.nan)
        weights = np.full(len(ids), np.nan)
        weighted = np.full(len(ids), np.nan)
        assigned = np.full(len(ids), -1, dtype=int)
        for number, band in enumerate(bands):
            indices = np.searchsorted(ids, band["source_ids"])
            assert np.array_equal(ids[indices], band["source_ids"])
            assert len(indices) == band["count"] and np.all(assigned[indices] == -1)
            mask = np.abs(radii-band["radius_km"]) <= band["tolerance_km"]
            assert np.array_equal(np.flatnonzero(mask), indices)
            band.update(angular_coverage(angles[indices]))
            scores[indices] = band["angular_score_deg_perigees"]
            if resolution_deg is not None:
                weights[indices] = angular_weights(angles[indices], resolution_deg)
                effective_count = float(weights[indices].sum())
                band.update(
                    effective_perigee_count=effective_count,
                    angular_resolution_deg=float(resolution_deg),
                    angular_score_weighted_deg_perigees=effective_count*band["angle_span_deg"],
                    crowding_factor=effective_count/band["count"],
                )
                weighted[indices] = band["angular_score_weighted_deg_perigees"]
            assigned[indices] = number
            # Tutti i punti devono stare nell'arco complemento del massimo gap.
            offsets = np.mod(angles[indices]-band["covered_arc_start_deg"], 360.0)
            assert np.all(offsets <= band["angle_span_deg"]+1e-10)
        assert np.isfinite(scores).all() and np.all(assigned >= 0)
        point_scores.append(scores)
        assignments.append(assigned)
        if resolution_deg is not None:
            assert np.isfinite(weights).all() and np.isfinite(weighted).all()
            point_weights.append(weights)
            weighted_scores.append(weighted)
    score_field = ("angular_score_weighted_deg_perigees" if resolution_deg is not None
                   else "angular_score_deg_perigees")
    maximum = max(b[score_field] for bs in report["bands"].values() for b in bs)
    report.update(
        angular_definition="span = 360 gradi - massimo gap fra angoli ordinati, incluso ultimo -> primo+360.",
        angular_score_definition="S = numero di perigei distinti nella fascia * span [gradi].",
        angular_frame="Angoli atan2(y,x) delle posizioni geocentriche sinodiche dei perigei.",
        color_definition="Cerchi/arco e pallini: stesso indice S della rispettiva fascia.",
        color_limits=[0.0, maximum], colormap="viridis", normalization="Lineare comune alle tre tolleranze",
        span_interpretation="Arco minimo contenente tutti i punti; non ogni angolo intermedio e campionato.",
        l1_geocentric_xy_km=[l1_position_km(), 0.0],
    )
    if resolution_deg is not None:
        report.update(
            angular_resolution_deg=float(resolution_deg),
            angular_score_definition="S_pesato = span * N_eff, con N_eff = somma dei pesi individuali.",
            weight_definition="w_i = 1 / somma_j max(0, 1-d_circolare(i,j)/risoluzione); include j=i.",
            color_definition="Cerchi/arco e pallini: stesso indice pesato della rispettiva fascia.",
        )
    data = dict(source_ids=ids, rp_km=radii, synodic_xy_km=points, angle_deg=angles,
                tolerances_km=np.array(list(report["bands"]), dtype=int),
                band_indices=np.array(assignments), angular_score_deg_perigees=np.array(point_scores))
    if resolution_deg is not None:
        data.update(angular_weights=np.array(point_weights),
                    angular_score_weighted_deg_perigees=np.array(weighted_scores),
                    angular_resolution_deg=float(resolution_deg))
    return data, report


def draw(tolerance, bands, data, scores, norm, resolution_deg=None):
    cmap = plt.get_cmap("viridis")
    weighted = resolution_deg is not None
    score_field = "angular_score_weighted_deg_perigees" if weighted else "angular_score_deg_perigees"
    colors = cmap(norm([b[score_field] for b in bands]))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([.055, .21, .48, .66])
    table_ax = fig.add_axes([.58, .17, .39, .70])
    title = (f"Copertura angolare pesata — risoluzione {resolution_deg:g}°" if weighted
             else "Copertura angolare × numero di perigei")
    formula = ("S = N_eff × span · perigei vicini hanno peso ridotto" if weighted
               else "S = N × (360° − massimo intervallo angolare vuoto)")
    fig.suptitle(f"{title} — tolleranza ±{int(tolerance):,} km\n{formula}", fontsize=17)
    for band, color in zip(bands, colors):
        radius, tol = band["radius_km"], band["tolerance_km"]
        start, end = band["covered_arc_start_deg"], band["covered_arc_end_unwrapped_deg"]
        ax.add_patch(Circle((0, 0), radius, fill=False, color="0.7",
                            linewidth=.7, linestyle="--", alpha=.7))
        if band["angle_span_deg"] > 0:
            ax.add_patch(Wedge((0, 0), radius+tol, start, end, width=2*tol,
                               facecolor=color, edgecolor="none", alpha=.09))
            ax.add_patch(Arc((0, 0), 2*radius, 2*radius, theta1=start, theta2=end,
                             color=color, linewidth=1.2, zorder=3))
    scatter = ax.scatter(*data["synodic_xy_km"].T, c=scores, cmap=cmap, norm=norm,
                         s=24, edgecolor="0.15", linewidth=.3, zorder=4)
    cax = fig.add_axes([.105, .115, .38, .018])
    bar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    bar.set_label("S = span angolare [°] × numero effettivo di perigei" if weighted
                  else "S = span angolare [°] × numero di perigei", fontsize=10)
    ax.add_patch(Circle((0, 0), R_EARTH_KM, color="royalblue", zorder=5))
    ax.annotate("Terra", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.plot(D_MOON_KM, 0, "o", color="0.3", markersize=4)
    ax.annotate("Luna", (D_MOON_KM, 0), xytext=(-5, 8),
                textcoords="offset points", ha="right", fontsize=9)
    draw_l1(ax)
    ax.set(xlabel="x geocentrica sinodica [km]", ylabel="y geocentrica sinodica [km]",
           xlim=(-410000, 410000), ylim=(-410000, 410000),
           title=f"{len(bands)} fasce disgiunte · {sum(b['count'] for b in bands)}/{len(scores)} perigei inclusi")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    table_ax.axis("off")
    table_ax.set_title("Raggio dal centro terrestre · span in gradi · S = N_eff × span" if weighted
                       else "Raggio dal centro terrestre · span in gradi · S = N × span", fontsize=11)
    chunks = np.array_split(np.arange(len(bands)), 2 if len(bands) > 30 else 1)
    for column, indices in enumerate(chunks):
        rows = []
        for i in indices:
            row = [str(i+1), f"{bands[i]['radius_km']:,.0f}", str(bands[i]["count"])]
            if weighted:
                row.append(f"{bands[i]['effective_perigee_count']:.1f}")
            row.extend([f"{bands[i]['angle_span_deg']:.1f}", f"{bands[i][score_field]:.1f}"])
            rows.append(row)
        labels = ["#", "R [km]", "N"] + (["N_eff"] if weighted else []) + ["Span [°]", "S"]
        widths = [.05, .27, .08, .16, .20, .24] if weighted else [.06, .32, .10, .23, .29]
        table = table_ax.table(cellText=rows, colLabels=labels,
                               cellLoc="center", colWidths=widths,
                               bbox=[column/len(chunks)+.01, .04, 1/len(chunks)-.03, .9])
        table.auto_set_font_size(False)
        table.set_fontsize(7.5 if len(chunks) == 2 else 11)
        for row, index in enumerate(indices, 1):
            for col in range(len(labels)):
                table[row, col].set_facecolor((*colors[index][:3], .16))
                table[row, col].set_edgecolor("0.9")
    footer = (f"Pesi ridotti per perigei vicini entro {resolution_deg:g}°; N_eff e la somma dei pesi. "
              "Pallini e arco hanno lo stesso S della fascia.\n"
              "Arco: span minimo contenente i punti; grigio: massimo gap vuoto. Tutti i 341 perigei inclusi; stessa scala nelle tre tolleranze."
              if weighted else
              "Arco colorato: span minimo contenente i perigei. Tratteggio grigio: massimo gap vuoto.\n"
              "Pallini e arco hanno lo stesso S della fascia; ogni traiettoria conta una volta. Stessa scala nelle tre tolleranze.")
    fig.text(.5, .035, footer,
             ha="center", va="center", fontsize=10.5)
    return fig


def main(resolution_deg=5.0):
    data, report = prepare(resolution_deg)
    destination = (DESTINATION / f"pesata_{resolution_deg:g}deg" if resolution_deg is not None else DESTINATION)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "perigei_angoli_e_indice.npz", **data)
    (destination / "riepilogo.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ["tolerance_km", "radius_km", "count", "largest_gap_deg", "angle_span_deg",
              "angular_score_deg_perigees", "largest_gap_start_deg", "largest_gap_end_deg",
              "covered_arc_start_deg", "covered_arc_end_unwrapped_deg"]
    if resolution_deg is not None:
        fields += ["angular_resolution_deg", "effective_perigee_count", "crowding_factor",
                   "angular_score_weighted_deg_perigees"]
    with (destination / "statistiche_angolari.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for bands in report["bands"].values():
            writer.writerows(bands)
    norm = Normalize(*report["color_limits"])
    score_field = "angular_score_weighted_deg_perigees" if resolution_deg is not None else "angular_score_deg_perigees"
    with PdfPages(destination / "copertura_angolare_tutte_tolleranze.pdf") as pdf:
        for row, (tolerance, bands) in enumerate(report["bands"].items()):
            fig = draw(tolerance, bands, data, data[score_field][row], norm, resolution_deg)
            fig.savefig(destination / f"copertura_angolare_tolleranza_{tolerance}km.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps(dict(output=str(destination), figures=len(report["bands"]),
                         color_limits=report["color_limits"], coverage=report["coverage_per_tolerance"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angular-resolution-deg", type=float, default=5.0,
                        help="Distanza di riferimento per ridurre la ridondanza angolare (default: 5 gradi).")
    parser.add_argument("--unweighted", action="store_true", help="Riproduce lo score originale N*span.")
    args = parser.parse_args()
    main(None if args.unweighted else args.angular_resolution_deg)
