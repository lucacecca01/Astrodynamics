"""Measure cluster dispersion of osculating semimajor axis and periapsis.

Uses individual saved eps/e values, without rerunning GA16 or integrations.
The signed semimajor axis is negative for hyperbolic orbits. Exact parabolic
members make the corresponding statistic undefined; they are never discarded.
"""

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "CR3BP_prova_GA_16_eps_e.py"


def model_mu_earth(source_script=SOURCE):
    """Read the actual model constants without executing the production main."""
    constants = {}
    for node in ast.parse(Path(source_script).read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"G", "m1", "mu_earth"}):
            value = compile(ast.Expression(node.value), str(source_script), "eval")
            constants[node.targets[0].id] = eval(value, {"__builtins__": {}}, constants)
    mu = float(constants["mu_earth"])
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Parametro gravitazionale terrestre non valido.")
    return mu


def model_earth_radius(source_script=SOURCE):
    for node in ast.parse(Path(source_script).read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "R_E"):
            radius = float(ast.literal_eval(node.value))
            if np.isfinite(radius) and radius > 0:
                return radius
    raise ValueError("Raggio terrestre del modello non valido.")


def individual_orbit_sizes(eps, eccentricity, mu_earth):
    eps, eccentricity = np.asarray(eps, dtype=float), np.asarray(eccentricity, dtype=float)
    if (eps.shape != eccentricity.shape or eps.ndim != 1
            or not np.isfinite(eps).all() or not np.isfinite(eccentricity).all()
            or np.any(eccentricity < 0) or not np.isfinite(mu_earth) or mu_earth <= 0):
        raise ValueError("Servono eps/e finiti e allineati e mu_earth positivo.")
    if np.any(((eps < 0) & (eccentricity >= 1)) | ((eps > 0) & (eccentricity <= 1))):
        raise ValueError("Energia ed eccentricita descrivono tipi di conica incompatibili.")
    a = np.full(eps.shape, np.nan)
    np.divide(-mu_earth, 2*eps, out=a, where=eps != 0)
    rp = a*(1-eccentricity)
    if np.any(np.isfinite(rp) & (rp <= 0)):
        raise ValueError("Raggio del perigeo non positivo.")
    return a, rp


def cluster_orbit_size_statistics(catalog, mu_earth, *, earth_radius_km=None):
    catalog = np.asarray(catalog)
    if catalog.ndim != 2 or catalog.shape[1] != 12 or not len(catalog):
        raise ValueError("Atteso il catalogo GA16 a 12 colonne.")
    if earth_radius_km is not None and (not np.isfinite(earth_radius_km) or earth_radius_km <= 0):
        raise ValueError("Raggio terrestre non valido.")
    labels = catalog[:, 1]
    if not np.isfinite(labels).all() or np.any(labels != np.floor(labels)) or np.any(labels < -1):
        raise ValueError("Etichette dei cluster non valide.")
    ordered = np.flatnonzero(labels >= 0)
    ordered = ordered[np.argsort(labels[ordered], kind="stable")]
    if not len(ordered):
        raise ValueError("Il catalogo non contiene cluster assegnati.")
    groups = np.split(ordered, np.flatnonzero(np.diff(labels[ordered]))+1)
    rows = []
    for group in groups:
        eps, eccentricity = catalog[group, 9], catalog[group, 10]
        a, rp = individual_orbit_sizes(eps, eccentricity, mu_earth)
        row = dict(cluster_id=int(labels[group[0]]), n_trajectories=len(group),
                   n_elliptic=int(np.count_nonzero(eps < 0)),
                   n_hyperbolic=int(np.count_nonzero(eps > 0)),
                   n_parabolic=int(np.count_nonzero(eps == 0)),
                   min_abs_eps_km2_s2=float(np.abs(eps).min()))
        for name, values in (("a", a), ("rp", rp)):
            finite = np.isfinite(values)
            row[f"n_defined_{name}"] = int(finite.sum())
            # Missing values are visible: never report a subset's STD as full-cluster STD.
            for statistic, function in (("mean", np.mean), ("std", np.std),
                                        ("min", np.min), ("max", np.max), ("median", np.median)):
                row[f"{statistic}_{name}_km"] = float(function(values)) if finite.all() else np.nan
        if earth_radius_km is not None:
            row["n_rp_below_earth_surface"] = int(np.count_nonzero(rp < earth_radius_km))
            row["min_h_rp_km"] = row["min_rp_km"]-earth_radius_km
        rows.append(row)
    return rows


def save_orbit_size_dispersion(catalog, output_directory, *, source_script=SOURCE):
    from matplotlib import pyplot as plt
    from ga16_recover_feature_plots import _save_and_close

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    mu = model_mu_earth(source_script)
    earth_radius = model_earth_radius(source_script)
    rows = cluster_orbit_size_statistics(catalog, mu, earth_radius_km=earth_radius)
    stem = f"orbital_size_dispersion_K{len(rows)}"
    path = output / f"{stem}.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    ids = np.array([row["cluster_id"] for row in rows])
    hyperbolic = np.array([row["n_hyperbolic"] > 0 for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(bottom=.20, top=.82, wspace=.25)
    for ax, quantity, label in zip(axes, ("a", "rp"),
                                    (r"STD $a$ [km]", r"STD $r_p$ [km]")):
        values = np.array([row[f"std_{quantity}_km"] for row in rows])
        for mask, color, description in ((~hyperbolic, "tab:blue", "Solo membri ellittici"),
                                          (hyperbolic, "tab:orange", "Include membri iperbolici")):
            if mask.any():
                ax.scatter(ids[mask], values[mask], color=color, s=22, label=description)
        positive = values[np.isfinite(values) & (values > 0)]
        if len(positive) == len(values) and positive.max()/positive.min() > 100:
            ax.set_yscale("log")
            ax.set_title("Scala logaritmica")
        else:
            ax.set_ylim(bottom=0)
        ax.set(xlabel="Cluster ID", ylabel=label)
        ax.grid(alpha=.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"Dispersione delle dimensioni orbitali al perigeo — K={len(rows)}\n"
                 "Deviazione standard della popolazione, calcolata sui singoli membri (ddof=0)")
    fig.text(.5, .035,
             r"$a=-\mu_{\oplus}/(2\varepsilon)$; $r_p=a(1-e)$. Elementi osculanti geocentrici; "
             "a negativo per le iperboli.\nNessun membro escluso. Valori e conteggi completi nel CSV.",
             ha="center", va="bottom", fontsize=10)
    _save_and_close(fig, output / f"{stem}.png")
    summary = dict(n_clusters=len(rows), n_assigned=sum(row["n_trajectories"] for row in rows),
                   n_unassigned=int(np.count_nonzero(catalog[:, 1] < 0)), mu_earth_km3_s2=mu,
                   earth_radius_km=earth_radius,
                   ddof=0, formulas=dict(a="-mu_earth/(2*eps)", rp="a*(1-e)"),
                   meaning="Earth osculating elements at the saved event; signed a, not abs(a).",
                   no_members_discarded=True, statistics={})
    for quantity in ("a", "rp"):
        values = np.array([row[f"std_{quantity}_km"] for row in rows])
        finite = np.isfinite(values)
        summary["statistics"][quantity] = dict(
            n_defined_clusters=int(finite.sum()),
            minimum_std_km=float(values[finite].min()) if finite.any() else None,
            median_std_km=float(np.median(values[finite])) if finite.any() else None,
            maximum_std_km=float(values[finite].max()) if finite.any() else None,
            maximum_std_cluster_id=int(ids[finite][np.argmax(values[finite])]) if finite.any() else None,
        )
    summary["n_clusters_including_hyperbolic_members"] = int(hyperbolic.sum())
    summary["n_members_rp_below_earth_surface"] = sum(row["n_rp_below_earth_surface"] for row in rows)
    summary["clusters_rp_below_earth_surface"] = [row["cluster_id"] for row in rows
                                                if row["n_rp_below_earth_surface"]]
    path_json = output / f"{stem}.json"
    temporary = path_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    temporary.replace(path_json)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "Clusters/GA_16_plots_eps0p02_e0p04_K196")
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    catalog = np.load(args.catalog or args.output / "trajectory_exit_catalog.npy", mmap_mode="r")
    print(json.dumps(save_orbit_size_dispersion(catalog, args.output), indent=2))


if __name__ == "__main__":
    main()
