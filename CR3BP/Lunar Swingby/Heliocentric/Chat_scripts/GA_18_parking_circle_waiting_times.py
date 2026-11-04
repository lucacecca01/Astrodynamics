"""Attesa massima e media per raggiungere una direzione di escape nel sinodico.

Parcheggio circolare kepleriano retrogrado e Luna circolare del catalogo.
Le fasce includono tutti i 341 perigei; le finestre temporali degli asteroidi
non entrano in questa stima cinematica della disponibilita rispetto alla Luna.
"""

import argparse
import csv
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import FuncFormatter, MaxNLocator

from GA_18_perigee_parking_circles import (
    CURVE_DATA, D_MOON_KM, OUTPUT, R_EARTH_KM, draw_l1, l1_position_km,
)


INPUT = OUTPUT / "estensione_radiale/asteroidi_vsoi"
DESTINATION = OUTPUT / "estensione_radiale/tempi_attesa"
MU_EARTH = 6.67430e-20 * 5.974e24
MU_MOON = 6.67430e-20 * 7.348e22
N_MOON_RAD_S = np.sqrt((MU_EARTH+MU_MOON)/D_MOON_KM**3)
PARKING_DIRECTION = -1  # Stesso verso retrogrado di tutte le escape selezionate.
CONTRAST_WINDOWS_HOURS = {"attesa_massima": (35.0, 55.0), "attesa_media": (6.0, 16.0)}
UNDER_COLOR = "#00B8D9"
OVER_COLOR = "#555555"
METRICS = (
    dict(key="attesa_massima", field="maximum_wait_hours", title="Attesa massima",
         subtitle="Peggior fase iniziale: tempo per raggiungere la prossima direzione utile"),
    dict(key="attesa_media", field="mean_wait_hours", title="Attesa media",
         subtitle="Media del tempo alla prossima direzione utile, su fase iniziale uniforme"),
)


def waiting_statistics(angles_deg, omega_relative_deg_hour):
    """Tempi diretti al prossimo punto: max gap/omega e sum(gap**2)/(720*omega)."""
    angles = np.asarray(angles_deg, dtype=float)
    if (angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all()
            or not np.isfinite(omega_relative_deg_hour)):
        raise ValueError("Servono angoli finiti e una velocita angolare relativa finita.")
    unique_angles = np.unique(np.mod(angles, 360.0))
    gaps = np.diff(np.append(unique_angles, unique_angles[0]+360.0))
    if len(unique_angles) == 1:
        gaps = np.array([360.0])
    speed = abs(float(omega_relative_deg_hour))
    return dict(
        largest_gap_deg=float(gaps.max()), angular_gaps_deg=gaps.tolist(),
        sum_squared_gaps_deg2=float(np.sum(gaps**2)),
        relative_angular_velocity_deg_hour=float(omega_relative_deg_hour),
        relative_cycle_hours=float(360.0/speed) if speed else float("inf"),
        maximum_wait_hours=float(gaps.max()/speed) if speed else float("inf"),
        mean_wait_hours=float(np.sum(gaps**2)/(720.0*speed)) if speed else float("inf"),
    )


def prepare():
    report = json.loads((INPUT / "riepilogo.json").read_text())
    with np.load(INPUT / "perigei_asteroidi_vsoi.npz") as data:
        ids, radii, points = (data[key] for key in ("source_ids", "rp_km", "synodic_xy_km"))
    assert np.array_equal(ids, np.unique(ids))
    assert np.allclose(np.linalg.norm(points, axis=1), radii, rtol=0, atol=1e-5)
    # Verifica il verso sui singoli membri, non dalla sola geometria dei plot.
    with np.load(CURVE_DATA / "selezione_istanze.npz") as data:
        members = data["members"]
    indices = np.searchsorted(members[:, 0], ids)
    assert np.array_equal(members[indices, 0], ids)
    features = np.load(CURVE_DATA / "reconstructed_features.npy", mmap_mode="r")
    assert np.all(features[indices, 3] == PARKING_DIRECTION), "Escape con versi diversi: separare i parcheggi."

    angles = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
    maximum_by_point, mean_by_point, assignments = [], [], []
    for bands in report["bands"].values():
        maximum = np.full(len(ids), np.nan)
        mean = np.full(len(ids), np.nan)
        assigned = np.full(len(ids), -1, dtype=int)
        for index, band in enumerate(bands):
            rows = np.searchsorted(ids, band["source_ids"])
            assert np.array_equal(ids[rows], band["source_ids"])
            assert len(rows) == band["count"] and np.all(assigned[rows] == -1)
            assert np.array_equal(np.flatnonzero(np.abs(radii-band["radius_km"]) <= band["tolerance_km"]), rows)
            parking_rate = PARKING_DIRECTION*np.sqrt(MU_EARTH/band["radius_km"]**3)
            relative_rate = np.degrees(parking_rate-N_MOON_RAD_S)*3600.0
            band.update(waiting_statistics(angles[rows], relative_rate))
            band["parking_angular_velocity_rad_s"] = float(parking_rate)
            assert 0 < band["mean_wait_hours"] <= band["maximum_wait_hours"]/2+1e-10
            maximum[rows] = band["maximum_wait_hours"]
            mean[rows] = band["mean_wait_hours"]
            assigned[rows] = index
        assert np.isfinite(maximum).all() and np.isfinite(mean).all() and np.all(assigned >= 0)
        assert sum(band["count"] for band in bands) == len(ids)
        maximum_by_point.append(maximum)
        mean_by_point.append(mean)
        assignments.append(assigned)
    all_times = [b[metric["field"]] for metric in METRICS
                 for bands in report["bands"].values() for b in bands]
    report.update(
        model="Stima cinematica: circolare kepleriana geocentrica retrograda e Luna circolare CR3BP.",
        parking_direction=PARKING_DIRECTION, perigee_directions_verified=len(ids),
        mu_earth_km3_s2=MU_EARTH, mu_moon_km3_s2=MU_MOON,
        moon_rate_rad_s=float(N_MOON_RAD_S),
        angular_frame="atan2(y,x) delle posizioni sinodiche geocentriche al perigeo.",
        omega_definition="omega_rel = -sqrt(mu_earth/R**3) - n_moon.",
        maximum_wait_definition="max(gap_deg) / abs(omega_rel_deg_per_hour).",
        mean_wait_definition="sum(gap_deg**2) / (720 * abs(omega_rel_deg_per_hour)); fase iniziale uniforme.",
        color_definition="Pallini e circolare: stessa statistica della fascia, non un tempo individuale a partire dal pallino.",
        assumptions="Raggi della fascia approssimati con quello della circolare. Finestre ETD/arrivo e raccordi dinamici non verificati qui.",
        color_limits_hours=[min(all_times), max(all_times)],
        normalization="Logaritmica, stessa scala in ore per entrambe le metriche e le tre tolleranze.",
        colormap="viridis_r", l1_geocentric_xy_km=[l1_position_km(), 0.0],
    )
    data = dict(source_ids=ids, rp_km=radii, synodic_xy_km=points, angle_deg=angles,
                tolerances_km=np.array(list(report["bands"]), dtype=int),
                band_indices=np.array(assignments), maximum_wait_hours=np.array(maximum_by_point),
                mean_wait_hours=np.array(mean_by_point), parking_direction=PARKING_DIRECTION)
    return data, report


def draw(metric, tolerance, bands, data, times, norm, focused=False, contrast=False):
    cmap = plt.get_cmap("RdYlBu_r" if contrast else "viridis_r").copy()
    if contrast:
        cmap.set_under(UNDER_COLOR)
        cmap.set_over(OVER_COLOR)
    colors = cmap(norm([band[metric["field"]] for band in bands]))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([.055, .21, .48, .66])
    show_profile = contrast and len(bands) <= 25
    table_ax = fig.add_axes([.58, .17, .39, .43 if show_profile else .70])
    fig.suptitle(f"{metric['title']} — parcheggio retrogrado — tolleranza ±{int(tolerance):,} km\n"
                 + metric["subtitle"], fontsize=16)
    for index, (band, color) in enumerate(zip(bands, colors)):
        radius, tol = band["radius_km"], band["tolerance_km"]
        above = contrast and band[metric["field"]] > norm.vmax
        ax.add_patch(Wedge((0, 0), radius+tol, 0, 360, width=2*tol,
                           facecolor=color, edgecolor="none", alpha=.03 if contrast else .07))
        ax.add_patch(Circle((0, 0), radius, fill=False, color=color,
                            linewidth=.8 if above else 2.0 if contrast else .9,
                            alpha=.6 if above else 1))
        if show_profile:
            angle = np.deg2rad(135 if radius < 60000 else 35 if index % 2 == 0 else 45)
            ax.annotate(str(index+1), (radius*np.cos(angle), radius*np.sin(angle)),
                        ha="center", va="center", fontsize=8.5, zorder=6,
                        bbox=dict(facecolor="white", edgecolor=color, linewidth=.7, alpha=.95, pad=1))
    scatter = ax.scatter(*data["synodic_xy_km"].T, c=times, cmap=cmap, norm=norm,
                         s=32 if contrast else 24, edgecolor="0.15", linewidth=.35 if contrast else .3, zorder=4)
    cax = fig.add_axes([.105, .115, .38, .018])
    bar = fig.colorbar(scatter, cax=cax, orientation="horizontal",
                       extend="both" if contrast else "max" if focused else "neither")
    if contrast:
        ticks = np.linspace(norm.vmin, norm.vmax, 6)
    elif focused:
        interior = MaxNLocator(nbins=4).tick_values(norm.vmin, norm.vmax)
        ticks = [norm.vmin]+[value for value in interior if norm.vmin < value < norm.vmax]+[norm.vmax]
    else:
        ticks = [value for value in (6, 12, 24, 48, 96, 192) if norm.vmin <= value <= norm.vmax]
    bar.set_ticks(ticks)
    bar.ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, position: f"{value:.1f}".rstrip("0").rstrip(".") if focused else f"{value:g}"))
    bar.set_label(f"Attesa [ore] · blu → rosso: {norm.vmin:g}–{norm.vmax:g} h" if contrast else
                  f"Attesa [ore] · scala lineare, satura oltre {norm.vmax:.1f} h" if focused
                  else "Attesa [ore] · scala logaritmica comune ai sei grafici", fontsize=10)
    ax.add_patch(Circle((0, 0), R_EARTH_KM, color="royalblue", zorder=5))
    ax.annotate("Terra", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.plot(D_MOON_KM, 0, "o", color="0.3", markersize=4)
    ax.annotate("Luna", (D_MOON_KM, 0), xytext=(-5, 8),
                textcoords="offset points", ha="right", fontsize=9)
    draw_l1(ax)
    ax.set(xlabel="x geocentrica sinodica [km]", ylabel="y geocentrica sinodica [km]",
           xlim=(-410000, 410000), ylim=(-410000, 410000),
           title=f"{len(bands)} fasce disgiunte · {sum(b['count'] for b in bands)}/{len(times)} perigei inclusi")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    table_ax.axis("off")
    table_ax.set_title("Raggio dal centro terrestre · massimo gap · attese in ore", fontsize=10 if show_profile else 11)
    if show_profile:
        profile = fig.add_axes([.615, .68, .35, .17])
        values = np.array([band[metric["field"]] for band in bands])
        radii = np.array([band["radius_km"] for band in bands])/1000
        inside = (values >= norm.vmin) & (values <= norm.vmax)
        profile.scatter(radii[inside], values[inside], c=values[inside], cmap=cmap,
                        norm=norm, s=32, edgecolor="black", linewidth=.4, zorder=3)
        if inside.any():
            xlo, xhi = radii[inside].min(), radii[inside].max()
            margin = max(5, .08*(xhi-xlo))
            profile.set_xlim(xlo-margin, xhi+margin)
        for index, (radius, value) in enumerate(zip(radii[inside], values[inside])):
            offset = 5 if index % 2 == 0 else -12
            if value < norm.vmin+.2*(norm.vmax-norm.vmin):
                offset = 5
            elif value > norm.vmax-.2*(norm.vmax-norm.vmin):
                offset = -12
            profile.annotate(f"{value:.2f}", (radius, value), xytext=(0, offset),
                             textcoords="offset points", ha="center", fontsize=7.5)
        profile.set(ylim=(norm.vmin, norm.vmax), xlabel="Raggio [migliaia di km]", ylabel="Ore",
                    title=f"Dettaglio {norm.vmin:g}–{norm.vmax:g} h: valori numerici")
        profile.set_title(profile.get_title(), fontsize=10)
        profile.xaxis.label.set_size(8)
        profile.yaxis.label.set_size(8)
        profile.tick_params(labelsize=8)
        profile.grid(alpha=.25)
    chunks = np.array_split(np.arange(len(bands)), 2 if len(bands) > 30 else 1)
    for column, indices in enumerate(chunks):
        rows = [[str(i+1), f"{bands[i]['radius_km']:,.0f}", str(bands[i]["count"]),
                 f"{bands[i]['largest_gap_deg']:.1f}", f"{bands[i]['maximum_wait_hours']:.1f}",
                 f"{bands[i]['mean_wait_hours']:.1f}"] for i in indices]
        table = table_ax.table(cellText=rows,
            colLabels=["#", "R [km]", "N", "Gap [°]", "Max [h]", "Media [h]"],
            cellLoc="center", colWidths=[.045, .26, .07, .18, .215, .23],
            bbox=[column/len(chunks)+.01, .04, 1/len(chunks)-.03, .9])
        table.auto_set_font_size(False)
        table.set_fontsize(7 if len(chunks) == 2 else 8.5 if show_profile and len(bands) > 14 else 10.5 if show_profile else 11)
        for row, index in enumerate(indices, 1):
            for col in range(6):
                active_column = 4 if metric["key"] == "attesa_massima" else 5
                alpha = (.42 if col == active_column else .18) if contrast else .16
                table[row, col].set_facecolor((*colors[index][:3], alpha))
                table[row, col].set_edgecolor("0.9")
                if contrast and col == active_column:
                    table[row, col].set_text_props(weight="bold")
    contrast_note = (
        "Numeri sui cerchi = righe della tabella. Profilo: dettaglio dell'intervallo cromatico; tempi reali sempre in tabella."
        if show_profile else
        "Righe ordinate per raggio; la colonna in grassetto indica la grandezza colorata. Tutti i tempi reali restano in tabella.")
    footer = (
        f"Blu → rosso: {norm.vmin:g}–{norm.vmax:g} h. Azzurro: sotto {norm.vmin:g} h; grigio: sopra {norm.vmax:g} h. Tutti i 341 perigei restano inclusi.\n"
        + contrast_note
        if contrast else
        f"Scala di dettaglio comune alle tre tolleranze: sopra {norm.vmax:.1f} h colore saturo; punta della barra = valori superiori. Tempi reali in tabella.\n"
        "Pallini e cerchio: statistica della fascia. Circolare retrograda kepleriana e Luna circolare; media su fase iniziale uniforme."
        if focused else
        "Pallini e cerchio: statistica della fascia. Media su fase iniziale uniforme. Giallo: attesa breve; viola: lunga.\n"
        "Circolare retrograda kepleriana e Luna circolare del catalogo; disponibilita geometrica delle escape, senza finestre d'arrivo GA18.")
    fig.text(.5, .035, footer,
             ha="center", va="center", fontsize=10.5)
    return fig


def main(focus_percentile=75.0, contrast=False):
    if focus_percentile is not None and (not np.isfinite(focus_percentile) or not 0 < focus_percentile < 100):
        raise ValueError("Il percentile della scala deve essere compreso fra 0 e 100.")
    data, report = prepare()
    focused = focus_percentile is not None or contrast
    destination = DESTINATION / "scala_contrasto" if contrast else DESTINATION / "scala_dettaglio" if focused else DESTINATION
    norms = {}
    if focused:
        report["raw_data_limits_hours"] = report.pop("color_limits_hours")
        report["color_scales"] = {}
        report["normalization"] = ("Lineare su intervallo di dettaglio fisso per metrica; stessa scala nelle tre tolleranze; colori separati fuori intervallo."
                                   if contrast else "Lineare per metrica; stessa scala nelle tre tolleranze; limite alto al percentile dei tempi associati ai perigei.")
        report["color_calibration"] = ("Finestra visiva 35–55 h per il massimo e 6–16 h per la media; dati completi invariati."
                                       if contrast else "Ogni source_id conta una volta per tolleranza: 341*3 valori; fasce pesate col loro numero di perigei.")
        report["upper_saturation_is_visual_only"] = True
        for metric in METRICS:
            values = data[metric["field"]].ravel()
            lower, upper = (CONTRAST_WINDOWS_HOURS[metric["key"]] if contrast
                            else (float(values.min()), float(np.percentile(values, focus_percentile))))
            if upper <= lower:
                raise ValueError("Il percentile non definisce un intervallo colore non degenere.")
            norms[metric["key"]] = Normalize(lower, upper, clip=False)
            report["color_scales"][metric["key"]] = dict(
                lower_hours=lower, upper_hours=upper,
                raw_maximum_hours=float(values.max()), calibration_values=len(values),
                saturated_perigee_instances=int(np.count_nonzero(values > upper)),
                saturated_bands_by_tolerance={tol: sum(band[metric["field"]] > upper for band in bands)
                                               for tol, bands in report["bands"].items()},
            )
            if contrast:
                report["color_scales"][metric["key"]].update(
                    under_perigee_instances=int(np.count_nonzero(values < lower)),
                    under_bands_by_tolerance={tol: sum(band[metric["field"]] < lower for band in bands)
                                               for tol, bands in report["bands"].items()},
                )
            else:
                report["color_scales"][metric["key"]]["upper_percentile"] = float(focus_percentile)
        if contrast:
            report.update(colormap="RdYlBu_r", under_color=UNDER_COLOR, over_color=OVER_COLOR,
                          outside_rendering_only=True)
    else:
        norms = {metric["key"]: LogNorm(*report["color_limits_hours"]) for metric in METRICS}
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "perigei_e_tempi_attesa.npz", **data)
    (destination / "riepilogo.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ["tolerance_km", "radius_km", "count", "largest_gap_deg", "sum_squared_gaps_deg2",
              "relative_angular_velocity_deg_hour", "relative_cycle_hours", "maximum_wait_hours", "mean_wait_hours"]
    with (destination / "tempi_attesa_fasce.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for bands in report["bands"].values():
            writer.writerows(bands)
    with PdfPages(destination / "attesa_massima_e_media_tutte_tolleranze.pdf") as all_pdf:
        for metric in METRICS:
            output = destination / metric["key"]
            output.mkdir(exist_ok=True)
            with PdfPages(output / "tutte_tolleranze.pdf") as metric_pdf:
                for row, (tolerance, bands) in enumerate(report["bands"].items()):
                    fig = draw(metric, tolerance, bands, data, data[metric["field"]][row], norms[metric["key"]], focused, contrast)
                    fig.savefig(output / f"tolleranza_{tolerance}km.png", dpi=180)
                    metric_pdf.savefig(fig)
                    all_pdf.savefig(fig)
                    plt.close(fig)
    print(json.dumps(dict(output=str(destination), figures=6,
                         color_scales=report.get("color_scales", report.get("color_limits_hours")),
                         moon_rate_deg_day=float(np.degrees(N_MOON_RAD_S)*86400.0),
                         coverage=report["coverage_per_tolerance"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-percentile", type=float,
                        help="Riproduce la scala al percentile richiesto, anziche le finestre di contrasto predefinite.")
    parser.add_argument("--full-scale", action="store_true", help="Riproduce la scala logaritmica completa originale.")
    parser.add_argument("--contrast", action="store_true", help="Usa la variante a contrasto con finestre strette e profili numerici.")
    args = parser.parse_args()
    main(None if args.full_scale else args.focus_percentile if args.focus_percentile is not None else 75.0,
         contrast=args.contrast and not args.full_scale)
