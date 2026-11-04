"""Confronta nove gradazioni e le risonanze nominali, tutte allineate alla Luna."""

import ast
import csv
import hashlib
import json
from math import gcd
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm, to_hex, to_rgba
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))


def model_configuration(source, cached_r0):
    """Carica le tre primarie senza creare o propagare la griglia completa."""
    text = source.read_text()
    tree = ast.parse(text)
    grid_position = next(
        k for k, node in enumerate(tree.body)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "distance"
    )
    grid = tree.body[grid_position]
    phase_loop = next(node for node in grid.body if isinstance(node, ast.For))
    namespace = {"__file__": str(source), "__name__": "_stability_examples_model"}
    exec(compile(ast.Module(body=tree.body[:grid_position], type_ignores=[]), str(source), "exec"), namespace)
    phases = eval(compile(ast.Expression(body=phase_loop.iter), str(source), "eval"), namespace)
    radii = namespace["distance_range"]
    if len(cached_r0) != len(radii) * len(phases):
        raise ValueError("La griglia corrente non corrisponde alla cache selezionata.")

    return namespace, text, tree, grid_position, phases


def nominal_resonances(namespace):
    """p:-q indica |n_particella|/n_Luna = p/q; posizioni kepleriane nominali."""
    moon_r = namespace["moon_x"] - namespace["earth_x"]
    moon_v = namespace["moon_v"] - namespace["earth_v"]
    earth, moon = namespace["M_EARTH"], namespace["M_MOON"]
    mu_em = namespace["sim"].G * (earth + moon)
    a_moon = 1 / (2 / np.linalg.norm(moon_r) - np.dot(moon_v, moon_v) / mu_em)
    lower, upper = np.min(namespace["distance_range"]), np.max(namespace["distance_range"])
    return sorted(
        (radius, p, q)
        for p in range(1, 6) for q in range(1, 6) if gcd(p, q) == 1
        for radius in [a_moon * (earth / (earth + moon))**(1/3) * (q/p)**(2/3)]
        if lower <= radius <= upper
    )


def initial_model(source, configuration, example_indices, resonances, cached_r0,
                  duration, sample_interval):
    """Ricostruisce solo gli esempi, mantenendo la configurazione originale."""
    namespace, text, tree, grid_position, phases = configuration
    radii = namespace["distance_range"]
    sim = namespace["sim"]
    for index in example_indices:
        radius, phase = radii[index // len(phases)], phases[index % len(phases)]
        sim.add(m=0, primary=sim.particles[1], a=radius, e=0,
                inc=np.pi, f=-phase, r=0)
        namespace["a_initial"].append(radius)

    for radius, _, _ in resonances:
        sim.add(m=0, primary=sim.particles[1], a=radius, e=0,
                inc=np.pi, f=-float(namespace["theta_moon"]), r=0)
        namespace["a_initial"].append(radius)

    # Si arresta prima degli array globali e della prima integrazione completa.
    setup = []
    for node in tree.body[grid_position + 1:]:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "n_particles"
            for target in node.targets
        ):
            break
        setup.append(node)
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(source), "exec"), namespace)
    # I colori appartengono alla finestra della cache, anche se Tmax e cambiato.
    namespace["source_duration"] = namespace["times"][-1] - sim.t
    sample_count = (int(round(duration / sample_interval)) + 1 if sample_interval is not None
                    else int(duration / namespace["dt"]))
    namespace["N"] = sample_count
    namespace["times"] = sim.t + np.linspace(0, duration, sample_count)
    namespace["Tmin"], namespace["Tmax"] = namespace["times"][[0, -1]]
    if sample_interval is not None and not np.isclose(
        namespace["times"][1] - namespace["times"][0], sample_interval, rtol=1e-10
    ):
        raise ValueError("Intervallo di campionamento della cache non ricostruibile.")
    namespace["a_initial"] = np.asarray(namespace["a_initial"])
    r0 = np.array([[p.x, p.y, p.z] for p in sim.particles[3:]])
    r0 -= np.array([sim.particles[1].x, sim.particles[1].y, sim.particles[1].z])
    if not np.allclose(r0[:len(example_indices)], cached_r0[example_indices], rtol=0, atol=1e-6):
        raise ValueError("Gli stati iniziali ricostruiti non corrispondono alla cache.")
    if not np.isclose(namespace["times"][-1] - sim.t, duration, rtol=0, atol=1e-6):
        raise ValueError("La durata corrente non corrisponde a quella della mappa.")
    return namespace, text, r0


def comparison_plot(output, filename, groups, titles, time_years, delta_a, e,
                    pericenter_angle, delta_r, colors, labels, stability_scores):
    """Scale uguali fra le colonne; ogni ordinata del semiasse e sempre a-a0."""
    fig, axes = plt.subplots(4, len(groups), figsize=(17, 12), sharex=True,
                             sharey="row", constrained_layout=True, squeeze=False)
    series = (delta_a, e, pericenter_angle, delta_r)
    ylabels = (r"$\Delta a=a(t)-a_0$ [km]", "Eccentricità",
               "Orientamento pericentro [°]", r"$|\Delta r|=|r(t)-r_0|$ [km]")
    outline = [pe.Stroke(linewidth=2.1, foreground="0.25", alpha=0.45), pe.Normal()]
    # S minore in primo piano; gli esempi scartati restano sullo sfondo.
    sort_scores = np.where(np.isfinite(stability_scores) & (stability_scores >= 0),
                           stability_scores, np.inf)
    zorders = np.empty(len(sort_scores), dtype=int)
    zorders[np.argsort(-sort_scores, kind="stable")] = np.arange(3, len(sort_scores) + 3)
    shown = np.concatenate(groups)
    for column, group in enumerate(groups):
        axes[0, column].set_title(titles[column])
        for j in group:
            for row, values in enumerate(series):
                if row == 2:
                    axes[row, column].scatter(time_years, values[:, j], color=colors[j],
                                              s=2, alpha=0.65, zorder=zorders[j])
                else:
                    axes[row, column].plot(time_years, values[:, j], color=colors[j], lw=1.2,
                                           label=labels[j] if row == 0 else None,
                                           path_effects=outline, zorder=zorders[j])
        axes[0, column].axhline(0, color="0.35", ls="--", lw=0.7)
        axes[3, column].axhline(0, color="0.35", ls="--", lw=0.7)
        legend = axes[0, column].legend(fontsize=7.6, loc="upper right")
        legend.set_zorder(len(sort_scores) + 4)
        for row in range(4):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].set_xlim(time_years[0], time_years[-1])
        axes[-1, column].set_xlabel("Tempo [anni]")
    for row, label in enumerate(ylabels):
        axes[row, 0].set_ylabel(label)
        if row in (1, 3):
            axes[row, 0].set_ylim(0, max(1e-10, 1.05 * np.nanmax(series[row][:, shown])))
        if row == 2:
            axes[row, 0].set_ylim(-180, 180)
    axes[0, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[3, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Tutti gli stati iniziali sulla semiretta Terra–Luna: fase relativa = 0°\n"
                 "Stessa scala di colori della mappa; risonanze indicate al semiasse nominale")
    fig.savefig(output / f"{filename}.png", dpi=200)
    fig.savefig(output / f"{filename}.pdf")
    plt.close(fig)


def main():
    source = SCRIPT_DIR / "Stability.py"
    cache = SCRIPT_DIR / "stability_periodicity_values.npz"
    if not cache.exists():
        cache = SCRIPT_DIR / "stability_megno_values.npz"
    output = SCRIPT_DIR / "esempi_allineati_luna"
    output.mkdir(exist_ok=True)

    with np.load(cache, allow_pickle=False) as data:
        r0 = data["r0"]
        saved_scores = data["stability_max"]
        removed = data["discarded_removed"].astype(bool)
        discarded_oe = data["discarded_oe"].astype(bool)
        duration = float(data["duration"])
        sample_interval = float(data["sample_interval"]) if "sample_interval" in data else None
    eligible = np.flatnonzero(np.isfinite(saved_scores) & (saved_scores >= 0))
    configuration = model_configuration(source, r0)
    namespace, source_text, source_tree, _, phases = configuration
    moon_r = namespace["moon_x"] - namespace["earth_x"]
    moon_angle = float(np.arctan2(moon_r[1], moon_r[0]))
    phase_offsets = (phases - moon_angle + np.pi) % (2 * np.pi) - np.pi
    zero_phase = int(np.argmin(np.abs(phase_offsets)))
    if abs(phase_offsets[zero_phase]) > 1e-10:
        raise ValueError("La griglia non contiene la fase relativa zero rispetto alla Luna.")
    aligned = eligible[eligible % len(phases) == zero_phase]
    if aligned.size < 9:
        raise ValueError("Servono almeno nove orbite valide sulla semiretta Terra–Luna.")

    # Stessa normalizzazione della mappa originale: non riscalare i nove esempi.
    norm_assignment = next(
        node for node in source_tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "norm" for target in node.targets)
    )
    vmax = float(saved_scores[eligible].max())
    norm = eval(compile(ast.Expression(body=norm_assignment.value), str(source), "eval"),
                {"PowerNorm": PowerNorm, "Normalize": plt.Normalize, "vmax": vmax})
    cmap = plt.get_cmap("RdYlGn_r")
    coordinates = norm(saved_scores[aligned])
    targets = np.linspace(float(coordinates.min()), float(coordinates.max()), 9)
    available = np.ones(aligned.size, dtype=bool)
    selected = []
    for target in targets:
        distances = np.where(available, np.abs(coordinates - target), np.inf)
        nearest = int(np.argmin(distances))
        selected.append(int(aligned[nearest]))
        available[nearest] = False
    selected = np.asarray(selected)
    resonances = nominal_resonances(namespace)
    namespace, model_source, example_r0 = initial_model(
        source, configuration, selected, resonances, r0, duration, sample_interval
    )
    n_gradations, n_examples = len(selected), len(selected) + len(resonances)
    ids = [f"G{j+1}" for j in range(n_gradations)] + [f"R{j+1}" for j in range(len(resonances))]
    resonance_labels = [""] * n_gradations + [f"{p}:−{q}" for _, p, q in resonances]
    original_indices = np.concatenate((selected, np.full(len(resonances), -1, dtype=int)))
    sim, times = namespace["sim"], namespace["times"]
    initial_a, yr, day = namespace["a_initial"], namespace["YR"], namespace["DAY"]
    phase = np.degrees(np.arctan2(example_r0[:, 1], example_r0[:, 0]))
    relative_phase = (phase - np.degrees(moon_angle) + 180) % 360 - 180
    if np.any(np.abs(relative_phase) > 1e-8) or np.any(example_r0 @ moon_r <= 0):
        raise ValueError("Un esempio non e allineato alla Luna sulla stessa semiretta.")
    r0_error = float(np.max(np.abs(example_r0[:n_gradations] - r0[selected])))
    print(f"Integrazione di {n_gradations} gradazioni + {len(resonances)} risonanze "
          f"allineate alla Luna per {duration / yr:g} anni...", flush=True)
    result = namespace["integration"](np.arange(3, sim.N), times)
    r_geo, v_geo, a, e, omega, computed_scores = result[1:]
    earth_distance = np.linalg.norm(r_geo, axis=2)
    delta_r = np.abs(earth_distance - earth_distance[0])
    if (not np.all(np.isfinite(r_geo[:, :n_gradations]))
            or not np.all(np.isfinite(v_geo[:, :n_gradations]))
            or not np.all(np.isfinite(computed_scores[:n_gradations]))
            or np.any(computed_scores[:n_gradations] < 0)):
        raise RuntimeError("Una reintegrazione non riproduce la classificazione originale.")
    delta_a = a - initial_a
    relative_a = delta_a / initial_a
    instantaneous_score = np.sqrt(e**2 + relative_a**2)
    score_error = np.abs(computed_scores[:n_gradations] - saved_scores[selected])
    color_error = np.abs(norm(computed_scores[:n_gradations]) - norm(saved_scores[selected]))
    if np.any(color_error > 1 / 256) or np.any(score_error > 1e-5):
        raise RuntimeError("I punteggi ricalcolati non riproducono i colori della mappa.")

    display_scores = computed_scores.copy()
    display_scores[:n_gradations] = saved_scores[selected]
    colors = cmap(norm(np.maximum(display_scores, 0)))
    colors[computed_scores < 0] = to_rgba("dimgray")
    incomplete = ~np.all(np.isfinite(a), axis=0)

    angular_momentum = np.cross(r_geo, v_geo)
    eccentricity_vector = np.cross(v_geo, angular_momentum) / namespace["mu_earth"]
    eccentricity_vector -= r_geo / np.linalg.norm(r_geo, axis=2)[..., None]
    np.testing.assert_allclose(np.linalg.norm(eccentricity_vector, axis=2), e,
                               rtol=1e-9, atol=1e-12, equal_nan=True)
    pericenter_angle = np.degrees(np.arctan2(eccentricity_vector[..., 1], eccentricity_vector[..., 0]))
    pericenter_angle[e < 1e-6] = np.nan
    t_years = times / yr
    rows = []
    labels = []
    for j, index in enumerate(original_indices):
        valid_times = np.flatnonzero(np.isfinite(a[:, j]))
        last_valid = int(valid_times[-1])
        status = ("rimossa per filtro lunare/collisioni" if incomplete[j]
                  else "scarto soglie OE" if computed_scores[j] < 0 else "intera finestra")
        header = ids[j] if j < n_gradations else f"{ids[j]} — {resonance_labels[j]} nominale"
        score_label = f"S={display_scores[j]:.4f}" if computed_scores[j] >= 0 else status
        labels.append(f"{header}: a₀={initial_a[j]:,.0f} km\n{score_label}")
        rows.append({
            "esempio": ids[j], "risonanza_nominale": resonance_labels[j],
            "indice_database_zero_based": int(index),
            "a0_km": float(initial_a[j]), "fase_iniziale_deg": float(phase[j]),
            "fase_relativa_luna_deg": float(relative_phase[j]),
            "stability_mappa": float(saved_scores[index]) if index >= 0 else None,
            "stability_ricalcolata": float(computed_scores[j]),
            "colore_hex": to_hex(colors[j]),
            "e_massima": float(np.nanmax(e[:, j])),
            "delta_a_min_km": float(np.nanmin(delta_a[:, j])),
            "delta_a_max_km": float(np.nanmax(delta_a[:, j])),
            "escursione_a_km": float(np.nanmax(a[:, j]) - np.nanmin(a[:, j])),
            "max_variazione_relativa_a": float(np.nanmax(np.abs(relative_a[:, j]))),
            "tempo_massimo_score_anni": float(t_years[np.nanargmax(instantaneous_score[:, j])]),
            "stato": status, "ultimo_campione_valido_giorni": float(times[last_valid] / day),
            "primo_campione_assente_giorni": (float(times[last_valid+1] / day) if incomplete[j] else None),
        })
    with (output / "esempi_selezionati.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "serie_OE_esempi.npz", times=times, time_years=t_years,
        example_ids=np.asarray(ids), resonance_labels=np.asarray(resonance_labels),
        original_indices=original_indices, initial_a_km=initial_a, initial_phase_deg=phase,
        initial_relative_phase_deg=relative_phase, initial_r_geo_km=example_r0,
        stability_saved=np.concatenate((saved_scores[selected], np.full(len(resonances), np.nan))),
        stability_recomputed=computed_scores, incomplete=incomplete,
        color_rgba=colors, r_geo=r_geo, v_geo=v_geo, a_km=a, e=e,
        delta_a_km=delta_a, relative_a=relative_a, instantaneous_score=instantaneous_score,
        pericenter_angle_deg=pericenter_angle, omega_code_convention_deg=omega,
        eccentricity_vector=eccentricity_vector,
    )

    comparison_plot(
        output, "confronto_OE_gradazioni", np.array_split(np.arange(n_gradations), 3),
        ("Verdi: G1–G3", "Gialli / arancio: G4–G6", "Rossi: G7–G9"),
        t_years, delta_a, e, pericenter_angle, delta_r, colors, labels, display_scores,
    )
    comparison_plot(
        output, "confronto_OE_risonanze", np.array_split(np.arange(n_gradations, n_examples), 3),
        ("Risonanze nominali: R1–R3", "Risonanze nominali: R4–R6", "Risonanze nominali: R7–R8"),
        t_years, delta_a, e, pericenter_angle, delta_r, colors, labels, display_scores,
    )

    # Una pagina per esempio, sempre con la variazione assoluta del semiasse.
    outline = [pe.Stroke(linewidth=2.1, foreground="0.25", alpha=0.45), pe.Normal()]
    with PdfPages(output / "OE_esempi_dettagliati.pdf") as pdf:
        for j, index in enumerate(original_indices):
            fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
            values = (delta_a[:, j], e[:, j], pericenter_angle[:, j], delta_r[:, j])
            axis_labels = (r"$\Delta a=a(t)-a_0$ [km]", "Eccentricità", "Orientamento pericentro [°]", r"$|\Delta r|=|r(t)-r_0|$ [km]")
            for k, (ax, value, label) in enumerate(zip(axes, values, axis_labels)):
                if k == 2:
                    ax.scatter(t_years, value, color=colors[j], s=3, edgecolors="0.3", linewidths=0.15)
                    ax.set_ylim(-180, 180)
                else:
                    ax.plot(t_years, value, color=colors[j], lw=1.2, path_effects=outline)
                ax.set_ylabel(label)
                ax.grid(alpha=0.25)
                if incomplete[j]:
                    first_missing = rows[j]["primo_campione_assente_giorni"] * day / yr
                    ax.axvspan(first_missing, t_years[-1], color="0.9", alpha=0.6)
                    ax.axvline(first_missing, color="0.35", ls=":", lw=0.8)
            axes[0].axhline(0, color="0.35", ls="--", lw=0.8, label=r"$\Delta a=0$")
            axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[0].legend(loc="upper right")
            axes[1].set_ylim(bottom=0)
            axes[3].axhline(0, color="0.35", ls="--", lw=0.8)
            axes[3].set_ylim(bottom=0)
            axes[3].ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[-1].set(xlabel="Tempo [anni]", xlim=(t_years[0], t_years[-1]))
            header = (f"{ids[j]} — indice mappa {index}" if j < n_gradations
                      else f"{ids[j]} — risonanza retrograda nominale {resonance_labels[j]}")
            fig.suptitle(f"{header}; {rows[j]['stato']}; S = {computed_scores[j]:.5f}\n"
                         f"a₀ = {initial_a[j]:,.1f} km; fase relativa Luna = 0° "
                         f"(azimut {phase[j]:.2f}°)\n"
                         "Pericentro: direzione geometrica del vettore eccentricità; omesso per e < 10⁻⁶")
            fig.savefig(output / f"OE_esempio_{ids[j]}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)

    # La seconda vista evita di sovrapporre etichette di punti quasi coincidenti.
    fig, (ax, ray) = plt.subplots(1, 2, figsize=(14, 9),
                                gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True)
    ax.scatter(r0[discarded_oe, 0], r0[discarded_oe, 1], c="lightgray", s=1, label="Scarti soglie OE")
    ax.scatter(r0[removed, 0], r0[removed, 1], c="dimgray", s=1, label="Filtro lunare / collisioni")
    ax.scatter(r0[eligible, 0], r0[eligible, 1], c=saved_scores[eligible], cmap=cmap, norm=norm, s=1)
    angle = np.linspace(0, 2*np.pi, 720)
    for radius, p, q in resonances:
        ax.plot(radius*np.cos(angle), radius*np.sin(angle), "k--", lw=0.7, alpha=0.6)
    moon_distance = np.linalg.norm(moon_r)
    ax.plot([0, moon_r[0]], [0, moon_r[1]], color="black", ls=":", lw=1.5,
            label="Fase relativa Luna = 0°", zorder=4)
    for group, marker, name in ((np.arange(n_gradations), "o", "Gradazioni G1–G9"),
                                 (np.arange(n_gradations, n_examples), "*", "Risonanze R1–R8")):
        ax.scatter(example_r0[group, 0], example_r0[group, 1], c=colors[group],
                   s=45 if marker == "o" else 95, marker=marker, edgecolors="black",
                   lw=0.7, zorder=5, label=name)
    for j in range(n_examples):
        is_grade = j < n_gradations
        x = 0 if is_grade else 1
        ray.scatter(x, initial_a[j], c=[colors[j]], s=50 if is_grade else 100,
                    marker="o" if is_grade else "*", edgecolors="black", zorder=5)
        ray.annotate(ids[j] if is_grade else f"{ids[j]} ({resonance_labels[j]})",
                     (x, initial_a[j]), xytext=(-8 if is_grade else 8, 0),
                     textcoords="offset points", ha="right" if is_grade else "left",
                     va="center", fontsize=9)
    ray.axhline(moon_distance, color="grey", ls="--", lw=1, label="Distanza iniziale Luna")
    ray.set(xlim=(-0.85, 2.1), xticks=[0, 1], xticklabels=["Gradazioni", "Risonanze"],
            ylabel="Distanza geocentrica iniziale [km]", title="Tutti sulla stessa semiretta")
    ray.grid(axis="y", alpha=0.25)
    ray.ticklabel_format(axis="y", style="plain", useOffset=False)
    ray.legend(fontsize=8, loc="upper left")
    ax.scatter(0, 0, c="blue", s=25, label="Terra", zorder=5)
    ax.scatter(moon_r[0], moon_r[1], c="grey", s=25, label="Luna iniziale", zorder=5)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 extend="max" if np.max(display_scores) > norm.vmax else "neither",
                 label=r"$S=\max_t\sqrt{e(t)^2+[(a(t)-a_0)/a_0]^2}$", shrink=0.75)
    ax.set(xlabel="X geocentrica [km]", ylabel="Y geocentrica [km]",
           title="Esempi allineati alla Luna\nCerchi: gradazioni; stelle: risonanze nominali")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output / "mappa_esempi.png", dpi=200)
    plt.close(fig)

    report = {
        "source": str(source), "source_sha256": hashlib.sha256(model_source.encode()).hexdigest(),
        "cache": str(cache), "number_of_integrated_test_particles": n_examples,
        "number_of_color_gradations": n_gradations, "number_of_nominal_resonances": len(resonances),
        "aligned_valid_conditions_in_cache": int(aligned.size),
        "duration_years": duration / yr, "samples": len(times),
        "source_duration_years": float(namespace["source_duration"] / yr),
        "time_grid_origin": "Durata e campionamento della cache della mappa selezionata",
        "initial_position_max_error_km": r0_error,
        "initial_relative_phase_max_absolute_deg": float(np.max(np.abs(relative_phase))),
        "nominal_resonance_caveat": "Posizione kepleriana nominale; la librazione non e verificata.",
        "semimajor_axis_plot_ordinate": "delta_a_km = a(t) - a0",
        "last_plot_ordinate": "abs_delta_r_km = abs(norm(r_geo, axis=2) - norm(r_geo[0], axis=1))",
        "color_overrange_examples": [ids[j] for j in range(n_examples) if display_scores[j] > norm.vmax],
        "score_max_absolute_error": float(score_error.max()),
        "color_coordinate_max_error": float(color_error.max()),
        "normalization": {"class": type(norm).__name__, "gamma": float(norm.gamma), "vmin": float(norm.vmin), "vmax": float(norm.vmax)},
        "examples": rows,
    }
    (output / "verifica_esempi.json").write_text(json.dumps(report, indent=2))
    if source.read_text() != model_source:
        raise RuntimeError("Stability.py e cambiato durante l'esecuzione degli esempi.")
    print(f"Grafici e serie temporali salvati in: {output}")
    print(f"Massimo errore sul punteggio originale: {score_error.max():.3g}")


if __name__ == "__main__":
    main()
