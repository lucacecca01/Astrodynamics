"""Conta i perigei della selezione individuale P99.99 in fasce circolari.

Legge risultati esistenti: nessuna integrazione, modifica dei cluster o Lambert.
Il conteggio misura la vicinanza geometrica, non valida nuovi raccordi dinamici.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from scipy.optimize import brentq


BASE = Path(__file__).resolve().parents[1] / "GA18_parking_orbits"
SELECTION = (BASE / "traiettorie_delta_v_leo200_hohmann_parcheggio"
             / "individuali/11_eps_phi/P99_99_selezione.npz")
CURVE_DATA = BASE / "traiettorie_phi"
OUTPUT = BASE / "parcheggi_circolari_perigei_341"
R_EARTH_KM = 6378.0
D_MOON_KM = 384400.0
EXTENDED_TOLERANCES_KM = (1000, 5000, 10000)


def l1_position_km():
    """L1 geocentrico nel CR3BP usato dal catalogo (Luna lungo +x)."""
    mass_ratio = 7.348e22 / (5.974e24 + 7.348e22)
    # q e la coordinata dalla Terra divisa per la distanza Terra-Luna;
    # la coordinata baricentrica che entra nel termine centrifugo e q-mu.
    def equilibrium(q):
        return q-mass_ratio-(1-mass_ratio)/q**2+mass_ratio/(1-q)**2
    return D_MOON_KM * brentq(equilibrium, 1e-8, 1-1e-8, xtol=1e-14)


def draw_l1(ax):
    position = l1_position_km()
    ax.plot(position, 0, marker="D", color="crimson", markersize=5,
            markeredgecolor="white", markeredgewidth=.6, zorder=7)
    ax.annotate("L1", (position, 0), xytext=(0, 12),
                textcoords="offset points", ha="center", color="crimson", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1), zorder=8)


def ranked_bands(radii, source_ids, tolerance, number=3):
    """Massimo esatto in finestre larghe 2*tol, poi picchi radiali separati.

    Fra finestre con uguale conteggio sceglie il centro piu basso. I centri
    successivi distano piu di 2*tol dai precedenti; non e un set-cover globale.
    Il centro scelto e il punto medio dei raggi estremi della finestra.
    """
    order = np.argsort(radii)
    sorted_radii = radii[order]
    candidates = []
    end = 0
    for start in range(len(radii)):
        while (end < len(radii)
               and sorted_radii[end] - sorted_radii[start] <= 2*tolerance):
            end += 1
        center = (sorted_radii[start]+sorted_radii[end-1])/2
        # Il centro puo includere anche raggi precedenti a "start".
        # Conta tutti i membri nella fascia effettiva prima di ordinarla.
        left = np.searchsorted(sorted_radii, center-tolerance, side="left")
        right = np.searchsorted(sorted_radii, center+tolerance, side="right")
        low, high = sorted_radii[left], sorted_radii[right-1]
        candidates.append((right-left, center, low, high))
    selected = []
    for count, center, low, high in sorted(candidates, key=lambda v: (-v[0], v[1])):
        if any(abs(center-b["radius_km"]) <= 2*tolerance for b in selected):
            continue
        mask = np.abs(radii-center) <= tolerance
        assert int(mask.sum()) == count
        selected.append(dict(
            tolerance_km=float(tolerance), radius_km=float(center),
            altitude_km=float(center-R_EARTH_KM), count=int(count),
            fraction=float(count/len(radii)),
            member_radius_range_km=[float(low), float(high)],
            admissible_center_range_km=[float(high-tolerance), float(low+tolerance)],
            source_ids=source_ids[mask].astype(int).tolist(),
        ))
        if number is not None and len(selected) == number:
            break
    return selected


def radial_counts(radii, centers, tolerance):
    sorted_radii = np.sort(radii)
    return (np.searchsorted(sorted_radii, centers+tolerance, side="right")
            - np.searchsorted(sorted_radii, centers-tolerance, side="left"))


def covering_bands(radii, source_ids, tolerance):
    """Copertura completa con il minimo numero di intervalli larghi 2*tol.

    Il primo raggio non coperto e il bordo interno della nuova fascia.
    Spingere il bordo esterno fino a r+2*tol non penalizza alcuna copertura
    successiva: il greedy ordinato e ottimo per questo problema geometrico.
    """
    radii = np.asarray(radii, dtype=float)
    source_ids = np.asarray(source_ids)
    assert radii.shape == source_ids.shape and np.isfinite(radii).all() and tolerance > 0
    order = np.argsort(radii)
    sorted_radii = radii[order]
    selected, start = [], 0
    covered = np.zeros(len(radii), dtype=bool)
    while start < len(radii):
        center = float(sorted_radii[start]+tolerance)
        # Protegge il punto sul bordo interno dall'arrotondamento di un ulp.
        if center-sorted_radii[start] > tolerance:
            center = float(np.nextafter(center, -np.inf))
        end = int(np.searchsorted(sorted_radii, center+tolerance, side="right"))
        assert end > start
        indices = order[start:end]
        mask = np.abs(radii-center) <= tolerance
        assert np.array_equal(np.flatnonzero(mask), np.sort(indices))
        assert not covered[indices].any()
        covered[indices] = True
        low, high = sorted_radii[start], sorted_radii[end-1]
        selected.append(dict(
            tolerance_km=float(tolerance), radius_km=center,
            altitude_km=float(center-R_EARTH_KM), count=int(end-start),
            fraction=float((end-start)/len(radii)),
            member_radius_range_km=[float(low), float(high)],
            admissible_center_range_km=[float(high-tolerance), float(low+tolerance)],
            source_ids=np.sort(source_ids[indices]).astype(int).tolist(),
        ))
        start = end
    assert covered.all() and sum(band["count"] for band in selected) == len(radii)
    return selected


def draw_extended(radii, source_ids, points, total_dv=None):
    """Copre tutti i perigei con fasce disgiunte, a tre tolleranze."""
    destination = OUTPUT / "estensione_radiale"
    dv_mode = total_dv is not None
    if dv_mode:
        destination = destination / "delta_v_totale"
        total_dv = np.asarray(total_dv, dtype=float)
        assert total_dv.shape == radii.shape and np.isfinite(total_dv).all()
        norm = Normalize(float(total_dv.min()), float(total_dv.max()))
        cmap = plt.get_cmap("viridis_r")
    destination.mkdir(parents=True, exist_ok=True)
    results = {
        str(tol): covering_bands(radii, source_ids, tol)
        for tol in EXTENDED_TOLERANCES_KM
    }
    if dv_mode:
        for bands in results.values():
            for band in bands:
                values = total_dv[np.abs(radii-band["radius_km"]) <= band["tolerance_km"]]
                band.update(
                    delta_v_total_mean_km_s=float(values.mean()),
                    delta_v_total_min_km_s=float(values.min()),
                    delta_v_total_max_km_s=float(values.max()),
                )
        np.savez_compressed(destination / "perigei_delta_v_totale.npz",
                            source_ids=source_ids, rp_km=radii, synodic_xy_km=points,
                            delta_v_total_km_s=total_dv)
    report = dict(
        selection=str(SELECTION), distinct_trajectories=len(radii),
        largest_perigee_radius_km=float(radii.max()),
        rule="Copertura completa ordinata: fascia [primo raggio non coperto, raggio+2*tolleranza].",
        note="Minimo numero di intervalli radiali di larghezza 2*tolleranza; ogni source_id conta una volta.",
        coverage_per_tolerance={tol: sum(band["count"] for band in bands) for tol, bands in results.items()},
        l1_geocentric_xy_km=[l1_position_km(), 0.0],
        bands=results,
    )
    if dv_mode:
        report.update(
            color_quantity="Hohmann LEO 200 km -> circolare(rp), piu immissione nell'escape",
            point_color="Delta V totale individuale", circle_color="Media nella fascia",
            normalization="Lineare comune a tutte le tolleranze, viridis_r",
            color_limits_km_s=[norm.vmin, norm.vmax],
            l1_geocentric_xy_km=[l1_position_km(), 0.0],
        )
    (destination / "conteggi_estesi.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ["tolerance_km", "radial_index", "radius_km", "altitude_km", "count", "fraction"]
    if dv_mode:
        fields += ["delta_v_total_mean_km_s", "delta_v_total_min_km_s", "delta_v_total_max_km_s"]
    with (destination / "conteggi_estesi.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for bands in results.values():
            for index, band in enumerate(bands, 1):
                writer.writerow({key: index if key == "radial_index" else band[key] for key in fields})

    with PdfPages(destination / "circolari_fino_perigei_esterni.pdf") as pdf:
        for tolerance, bands in results.items():
            covered = np.zeros(len(radii), dtype=bool)
            colors = (cmap(norm([band["delta_v_total_mean_km_s"] for band in bands]))
                      if dv_mode else plt.get_cmap("turbo")(np.linspace(.08, .9, len(bands))))
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_axes([.055, .21 if dv_mode else .17, .48, .66 if dv_mode else .70])
            table_ax = fig.add_axes([.58, .17, .39, .70])
            fig.suptitle(
                f"Circolari fino ai perigei piu esterni — tolleranza ±{int(tolerance):,} km\n"
                f"{len(radii)} perigei distinti · raggio massimo {radii.max():,.0f} km", fontsize=17)
            if not dv_mode:
                ax.scatter(*points.T, s=16, color="0.65", linewidths=0, zorder=3)
            for band, color in zip(bands, colors):
                radius, tol = band["radius_km"], band["tolerance_km"]
                mask = np.abs(radii-radius) <= tol
                assert not np.any(covered & mask), "Perigeo contato in due fasce."
                covered |= mask
                ax.add_patch(Wedge((0, 0), radius+tol, 0, 360, width=2*tol,
                                   facecolor=color, alpha=.06, edgecolor="none"))
                ax.add_patch(Circle((0, 0), radius, fill=False, color=color, linewidth=.8))
                if not dv_mode:
                    ax.scatter(*points[mask].T, s=22, color=color,
                               edgecolor="white", linewidth=.25, zorder=4)
            if dv_mode:
                # Anche i punti fuori dalle fasce mantengono il proprio costo.
                scatter = ax.scatter(*points.T, s=24, c=total_dv, cmap=cmap, norm=norm,
                                     edgecolor="0.15", linewidth=.3, zorder=4)
                cax = fig.add_axes([.105, .115, .38, .018])
                bar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
                bar.set_label("ΔV totale di partenza da LEO 200 km [km/s]", fontsize=10)
            assert covered.all(), "Almeno un perigeo e rimasto fuori dalle fasce."
            ax.add_patch(Circle((0, 0), R_EARTH_KM, color="royalblue", zorder=5))
            ax.annotate("Terra", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=9)
            ax.plot(D_MOON_KM, 0, "o", color="0.3", markersize=4)
            ax.annotate("Luna", (D_MOON_KM, 0), xytext=(-5, 8),
                        textcoords="offset points", ha="right", fontsize=9)
            draw_l1(ax)
            ax.set(xlabel="x geocentrica sinodica [km]", ylabel="y geocentrica sinodica [km]",
                   xlim=(-410000, 410000), ylim=(-410000, 410000),
                   title=f"{len(bands)} fasce disgiunte · {covered.sum()}/{len(radii)} perigei inclusi")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
            table_ax.axis("off")
            table_ax.set_title("Raggi dal centro terrestre, ordinati dall'interno all'esterno", fontsize=11)
            chunks = np.array_split(np.arange(len(bands)), 2 if len(bands) > 30 else 1)
            for column, indices in enumerate(chunks):
                rows = [[str(i+1), f"{bands[i]['radius_km']:,.0f}", str(bands[i]["count"])] for i in indices]
                if dv_mode:
                    for row, index in zip(rows, indices):
                        row.append(f"{bands[index]['delta_v_total_mean_km_s']:.3f}")
                table = table_ax.table(
                    cellText=rows,
                    colLabels=["#", "Raggio [km]", "Perigei"] + (["ΔV [km/s]"] if dv_mode else []),
                    cellLoc="center", colWidths=[.10, .41, .19, .30] if dv_mode else [.15, .57, .28],
                    bbox=[column/len(chunks)+.01, .04, 1/len(chunks)-.03, .9])
                table.auto_set_font_size(False)
                table.set_fontsize((8 if dv_mode else 9) if len(chunks) == 2 else 12)
                for row, index in enumerate(indices, 1):
                    for col in range(4 if dv_mode else 3):
                        cell = table[row, col]
                        cell.set_facecolor((*colors[index][:3], .16))
                        cell.set_edgecolor("0.9")
            footer = (
                "Pallini: ΔV individuale. Cerchi e tabella: ΔV medio dei perigei nella fascia. Stessa scala per tutte le tolleranze.\n"
                "ΔV totale = due impulsi Hohmann dalla LEO a 200 km alla circolare(rₚ) + immissione nell'escape; stesso piano e verso."
                if dv_mode else
                "Copertura completa dal perigeo piu interno al piu esterno, con il minimo numero di fasce radiali.\n"
                "Tutti i perigei sono inclusi; ogni punto conta una volta, indipendentemente dalle sue epoche ETD.\n"
                "Vicinanza geometrica: velocita e tempi di un eventuale raccordo richiedono una verifica separata.")
            fig.text(.5, .035 if dv_mode else .08, footer,
                     ha="center", va="center", fontsize=11)
            fig.savefig(destination / f"circolari_estese_tolleranza_{tolerance}km.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    return results


def main(color_by_total_dv=False):
    with np.load(SELECTION) as data:
        source_ids, first = np.unique(data["source_ids"], return_index=True)
        radii = data["rp_km"][first]
        assert data["event_is_periapsis"].all()
        # Ogni istanza ETD dello stesso membro ha lo stesso raggio.
        assert np.array_equal(data["rp_km"], radii[np.searchsorted(source_ids, data["source_ids"])])
        if color_by_total_dv:
            total_dv = data["delta_v_total_km_s"][first]
            assert np.array_equal(data["delta_v_total_km_s"],
                                  total_dv[np.searchsorted(source_ids, data["source_ids"])])
            assert np.allclose(data["delta_v_total_km_s"],
                               data["delta_v_injection_km_s"]+data["delta_v_hohmann_km_s"],
                               rtol=0, atol=1e-12)
            assert float(data["leo_altitude_km"]) == 200
    with np.load(CURVE_DATA / "selezione_istanze.npz") as data:
        members = data["members"]
    rows = np.searchsorted(members[:, 0], source_ids)
    assert np.array_equal(members[rows, 0], source_ids)
    curves = np.load(CURVE_DATA / "reconstructed_curves.npy", mmap_mode="r")
    points = np.asarray(curves[rows, :, 0])
    error = float(np.max(np.abs(np.linalg.norm(points, axis=1)-radii)))
    assert error < 1e-5, "Il punto iniziale non coincide con il periapside osculatore."
    if color_by_total_dv:
        results = draw_extended(radii, source_ids, points, total_dv=total_dv)
        print(json.dumps(dict(
            output=str(OUTPUT / "estensione_radiale/delta_v_totale"),
            delta_v_range_km_s=[float(total_dv.min()), float(total_dv.max())],
            bands={tol: len(bands) for tol, bands in results.items()},
        ), indent=2))
        return

    tolerances = (100, 500, 1000, 2500, 5000, 10000)
    results = {str(tol): ranked_bands(radii, source_ids, tol) for tol in tolerances}
    primary = results["1000"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = dict(
        selection=str(SELECTION), distinct_trajectories=len(radii),
        distinct_exact_radii=len(np.unique(radii)),
        geometry_radius_max_error_km=error, radius_origin="Centro della Terra",
        condition="abs(r_perigee - radius) <= tolerance_km",
        selection_rule="Picchi scelti in ordine di conteggio, con fasce disgiunte; non set-cover globale.",
        warning="Vicinanza geometrica: non sono nuove connessioni dinamiche validate.",
        bands=results,
    )
    (OUTPUT / "conteggi.json").write_text(json.dumps(report, indent=2)+"\n")
    with (OUTPUT / "conteggi.csv").open("w", newline="") as stream:
        fields = ["tolerance_km", "rank", "radius_km", "altitude_km", "count", "fraction"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for bands in results.values():
            for rank, band in enumerate(bands, 1):
                writer.writerow({key: rank if key == "rank" else band[key] for key in fields})

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.065, right=.975, bottom=.18, top=.87, wspace=.25)
    fig.suptitle(f"Circolari vicine al maggior numero di perigei\n"
                 f"{len(radii)} traiettorie distinte della selezione individuale P99.99", fontsize=17)
    ax = axes[0]
    ax.scatter(*points.T, s=17, color="0.7", linewidths=0, label="Altri perigei", zorder=3)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for band, color in zip(primary, colors):
        radius, tol = band["radius_km"], band["tolerance_km"]
        mask = np.abs(radii-radius) <= tol
        ax.add_patch(Wedge((0, 0), radius+tol, 0, 360, width=2*tol,
                           facecolor=color, alpha=.16, edgecolor="none"))
        ax.add_patch(Circle((0, 0), radius, fill=False, color=color, linewidth=1.5,
                           label=f"r = {radius:,.0f} km: {band['count']} perigei"))
        ax.scatter(*points[mask].T, s=28, color=color, edgecolor="white", linewidth=.3, zorder=4)
    ax.add_patch(Circle((0, 0), R_EARTH_KM, color="royalblue", label="Terra", zorder=5))
    ax.plot(D_MOON_KM, 0, "o", color="0.4", markersize=4)
    ax.annotate("Luna", (D_MOON_KM, 0), xytext=(-5, 8),
                textcoords="offset points", ha="right", fontsize=10)
    ax.set(xlabel="x geocentrica sinodica [km]", ylabel="y geocentrica sinodica [km]",
           title="Tre picchi separati con tolleranza ±1.000 km",
           xlim=(-410000, 410000), ylim=(-410000, 410000))
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=9, framealpha=.95)

    ax = axes[1]
    for tolerance, color in ((1000, "tab:blue"), (5000, "tab:red")):
        critical = np.unique(np.concatenate((radii-tolerance, radii+tolerance)))
        # Conteggio esatto su ciascun intervallo fra i cambiamenti di appartenenza.
        centers = (critical[:-1]+critical[1:])/2
        counts = radial_counts(radii, centers, tolerance)
        ax.stairs(counts, critical/1000, baseline=0, color=color, linewidth=1.1,
                  label=f"Tolleranza ±{tolerance:,.0f} km")
        best = results[str(tolerance)][0]
        ax.plot(best["radius_km"]/1000, best["count"], "o", color=color)
        ax.annotate(f"{best['count']} perigei a {best['radius_km']/1000:.1f} mila km",
                    (best["radius_km"]/1000, best["count"]), xytext=(15, 10),
                    textcoords="offset points", color=color, fontsize=10)
    for band, color in zip(primary, colors):
        ax.plot(band["radius_km"]/1000, band["count"], "o", color=color, zorder=5)
    ax.set(xlabel="Raggio della circolare dal centro terrestre [migliaia di km]",
           ylabel="Numero di traiettorie distinte nella fascia",
           title="Conteggio radiale senza binning", xlim=(0, 390), ylim=(0, 52))
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=10)
    fig.text(.5, .065,
             "Ogni traiettoria conta una sola volta, indipendentemente dalle sue epoche ETD.\n"
             "Si contano perigei con |rₚ − r| ≤ tolleranza: le fasce non sono intersezioni esatte.\n"
             "Una circolare comune richiede anche la verifica di velocità e sincronizzazione.",
             ha="center", va="center", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"circolari_perigei.{suffix}", dpi=180)
    plt.close(fig)
    extended = draw_extended(radii, source_ids, points)
    print(json.dumps({"distinct_perigees": len(radii), "output": str(OUTPUT),
                      "best": {tol: bands[0]["count"] for tol, bands in results.items()},
                      "extended_bands": {tol: len(bands) for tol, bands in extended.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-by-total-dv", action="store_true",
                        help="Ricolora i perigei col DV totale; cerchi col DV medio nella fascia.")
    main(color_by_total_dv=parser.parse_args().color_by_total_dv)
