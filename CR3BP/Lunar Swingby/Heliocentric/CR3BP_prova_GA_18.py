from Celestial_Mechanics import Integration, Transformations
import numpy as np
from pathlib import Path
from multiprocessing import get_context
import time
from matplotlib import pyplot as plt
import pykep as pk
from scipy.integrate import solve_ivp
from scipy.optimize import root, differential_evolution
import spiceypy as spice
import time


start_time = time.perf_counter()


# Define constants
G = 6.67430e-20              # Gravitational constant in km^3 kg^-1 s^-2
d = 384400                   # Average distance between Earth and Moon in km
m1 = 5.974e24                # Mass of Earth in kg
m2 = 7.348e22                # Mass of Moon in kg
ms = 1.989e30                # Mass of Sun in kg
d_E = 1.496e8                # Average distance between Earth and Sun in km
R_E = 6378                   # Earth radius in km
R_L = 1737                   # Moon radius in km
day = 86400                  # Day in s
M = m1 + m2                  # Total mass of the Earth-Moon system in kg
mu = m2/M                    # Moon-to-total mass ratio
n = np.sqrt(G * M / d**3)    # Moon’s mean motion
TU = 1 / n                   # Moon’s mean revolution period
day = 86400                  # Seconds in a day
E_SOI = (m1 / ms)**(2/5) * d_E
M_SOI = (m2 / m1)**(2/5) * d
mu_earth = G * m1
mu_sun = G * ms
n_earth = np.sqrt(mu_sun / d_E**3)
n_moon = n




# Output control flags
PLOT = True
SAVE = True
PRINT = True



# Define earth-moon geometry function
def earth_moon_geometry(jd):

    jd = np.asarray(jd, dtype=float)
    et = (jd - 2451545.0) * day
   
    def state(body, observer):
        values = np.array([spice.spkezr(body, float(epoch), "ECLIPJ2000", "NONE", observer)[0] for epoch in et.ravel()]).reshape(jd.shape + (6,))
        values[..., 2] = 0
        values[..., 5] = 0
        return values

    earth = state("EARTH", "SUN")
    moon_geo = state("MOON", "EARTH")

    theta_moon = np.arctan2(moon_geo[..., 1], moon_geo[..., 0])

    return (earth[..., :3], earth[..., 3:], earth[..., :3] + moon_geo[..., :3], theta_moon)



# Define function to convert escape velocities to heliocentric frame
def escape_velocity_to_heliocentric(velocities_catalog, v_earth, theta_moon):

    angle = theta_moon

    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])

    return (velocities_catalog @ rotation.T + np.asarray(v_earth)[..., :2])



# Define function to compute analytic TOF intervals for a given target
def target_tof_intervals(r0, r1, v_min, v_max, mu):

    r0, r1 = r0[:2], r1[:2]
    R0, R1 = np.linalg.norm(r0), np.linalg.norm(r1)

    cross = r0[0] * r1[1] - r0[1] * r1[0]

    vmax2 = np.sum(np.maximum(abs(v_min), abs(v_max))**2)

    chord = np.linalg.norm(r1 - r0)
    s = (R0 + R1 + chord) / 2
    lam = np.sqrt(max(0.0, 1 - chord / s))

    if cross < 0:
        lam = -lam

    rho = (R0 - R1) / chord
    sigma = np.sqrt(max(0.0, 1 - rho**2))
    er = r0 / R0
    et = np.array([-er[1], er[0]])
    scale = np.sqrt(mu * s / 2) / R0

    A = scale * (lam * (1 - rho) * er + sigma * et)
    B = scale * (-(1 + rho) * er + sigma * lam * et)

    xmax2 = 1 - s / R0 + s * vmax2 / (2 * mu)

    if xmax2 < -1e-13:
        return []
    
    xmax = np.sqrt(max(0.0, xmax2))
    tol_v = 1e-10 * max(1.0, np.sqrt(vmax2))

    def velocity(x):
        return A * np.sqrt(1 - lam**2 + lam**2 * x**2) + B * x

    def inside(x, tolerance=tol_v):
        v = velocity(x)
        return (np.all(v >= v_min - tolerance) and np.all(v <= v_max + tolerance))

    def tof(x):
        q = 1 - x**2
        y = np.sqrt(1 - lam**2 * q)
        psi = np.arccos(np.clip(x * y + lam * q, -1.0, 1.0))
        return ((psi / np.sqrt(q) - x + lam * y) / q * np.sqrt(s**3 / (2 * mu)))

    cuts = [-xmax, xmax]

    for j in range(2):
        for bound in (v_min[j], v_max[j]):
            roots = np.roots([A[j]**2 * lam**2 - B[j]**2, 2 * bound * B[j], A[j]**2 * (1 - lam**2) - bound**2])
            for z in roots:
                if abs(z.imag) <= 1e-7 and -xmax - 1e-12 <= z.real <= xmax + 1e-12:
                    x = np.clip(z.real, -xmax, xmax)
                    if abs(velocity(x)[j] - bound) <= tol_v:
                        cuts.append(x)

    cuts = np.unique(cuts)
    spans = [(x, x) for x in cuts if inside(x)]
    spans += [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if inside((a + b) / 2, tolerance=0)]

    merged = []

    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    return sorted((tof(b), tof(a)) for a, b in merged)





# Define parameters
N_SIGMA = 2.5                      # Number of standard deviations for the interval
N_TARGETS = 10000
SEED = 42
R_MIN_AU = 0.6
R_MAX_AU = 1.4





# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Clustering/Clusters/GA_16_plots_vinf_time_K605/cluster_velocity_statistics.txt"
data = np.loadtxt(DATA_FILE, ndmin=2)
spice.furnsh(pk.udpla.de440s.kernel_file())

cluster_ids = data[:, 0].astype(int)
cluster_sizes = data[:, 1].astype(int)

mean_v_soi = data[:, 2:4]
std_v_soi = data[:, 4:6] 

t_perigee = data[:, 6]
std_t_perigee = data[:, 8]

t_soi = data[:, 7]
std_t_soi = data[:, 9]

mean_r_soi = data[:, 10:12]
std_r_soi = data[:, 12:14]



radius_v = N_SIGMA * np.linalg.norm(std_v_soi, axis=1)

if PRINT:
    print(f"\nCluster caricati: {len(cluster_ids)}")
    print(f"Traiettorie rappresentate: {cluster_sizes.sum()}")
    print(f"Limite radiale: {N_SIGMA:g} sigma")





# Generate asteroid orbital elements
N_SAMPLE = 10000
TARGET_STEP = 0.01

transformations = Transformations()

t_targets_days = []
nu_targets = []
r_targets = []
asteroid_elements = []

rng = np.random.default_rng(SEED)

for _ in range(N_SAMPLE):
    a = rng.uniform(0.1, 10) * d_E            # Semimajor axis [km]
    e = rng.uniform(0.01, 0.9)                # Eccentricity []
    i = 0                                     # Inclination [rad]
    W = 0                                     # Ascending node [rad]
    w = rng.uniform(0, 2 * np.pi)             # Argument of perihelion [rad]
    M0 = rng.uniform(0, 2 * np.pi)            # Mean anomaly [rad]

    rp = a * (1 - e)
    ra = a * (1 + e)

    if ra < R_MIN_AU * d_E or rp > R_MAX_AU * d_E:
        continue

    elements_ast = [a, e, i, W, w, M0]

    nu_in = np.arccos(np.clip((a * (1 - e**2) / (R_MIN_AU * d_E) - 1) / e, -1, 1))
    nu_out = np.arccos(np.clip((a * (1 - e**2) / (R_MAX_AU * d_E) - 1) / e, -1, 1))

    nu_target = np.concatenate((np.arange(nu_in, nu_out, TARGET_STEP), np.arange(2*np.pi - nu_out, 2*np.pi - nu_in, TARGET_STEP),))

    if nu_target.size == 0:
        continue

    nu_0 = 0

    M_target = transformations.M_from_TA(nu_target, e)
    n_ast = np.sqrt(mu_sun / a**3)
    time_target_days = np.mod(M_target - M0, 2*np.pi) / n_ast / day

    nu_targets.append(nu_target)
    r_targets.append(np.array([pk.par2ic([*elements_ast[:5], nu], mu_sun)[0] for nu in nu_target]))
    asteroid_elements.append(elements_ast)
    t_targets_days.append(time_target_days)





# Compute heliocentric velocities
TARGET_ID = 7

elements = asteroid_elements[TARGET_ID]
a_ast, e_ast, _, _, _, M0_ast = elements
n_ast = np.sqrt(mu_sun / a_ast**3)

JD0 = 2461041.5                  # Epoca di riferimento: 1 gennaio 2026, 00:00 TDB
JD_START = JD0
JD_END = JD0 + 4 * np.pi / n_ast / day

transfer_window = JD_END - JD_START

STEP_ETD_DAYS = 1
STEP_ARRIVAL_DAYS = 1

jd_etd_grid = JD_START + np.append(np.arange(0, transfer_window, STEP_ETD_DAYS), transfer_window)
jd_arr_grid = JD_START + np.append(np.arange(0, transfer_window, STEP_ARRIVAL_DAYS), transfer_window)
   




# Analytic TOF intervals for every target, converted to days
M_arr = transformations.M_from_t(M0_ast, n_ast, (jd_arr_grid - JD0) * day, 0)
nu_arr = np.array([transformations.TA_from_M(float(M_phase), e_ast) for M_phase in np.mod(M_arr, 2*np.pi)])

arrival_targets = np.array([pk.par2ic([*elements[:5], float(nu)], mu_sun)[0] for nu in nu_arr])
radii = np.linalg.norm(arrival_targets, axis=1)

inside = ((radii >= R_MIN_AU*d_E) & (radii <= R_MAX_AU*d_E))
jd_targets = jd_arr_grid[inside]
arrival_targets = arrival_targets[inside]

arrival_velocities = np.array([pk.par2ic([*elements[:5], float(nu)], mu_sun)[1] for nu in nu_arr[inside]])





# Solve Heliocentric Lambert transfer
matches = []
tof_days_grid = []
lambert_velocities = []
lambert_compatible = []
lambert_cluster_rows = []

lambert_departures = []
lambert_arrivals = []
lambert_etd_indices = []
lambert_target_indices = []
rendezvous_dv = []

n_lambert = 0


for i_etd, jd_etd in enumerate(jd_etd_grid):

    jd_soi_current = jd_etd + t_soi

    theta_current = earth_moon_geometry(jd_etd)[3]
    r_earth_current, v_earth_current = earth_moon_geometry(jd_soi_current)[:2]

    rotation_current = np.array([[np.cos(theta_current), -np.sin(theta_current)],
                                 [np.sin(theta_current),  np.cos(theta_current)]])

    departures_current = r_earth_current.copy()
    departures_current[:, :2] += mean_r_soi @ rotation_current.T

    vinf_current = escape_velocity_to_heliocentric(mean_v_soi, v_earth_current, theta_current,)

    for j, (jd_arrival, r_target) in enumerate(zip(jd_targets, arrival_targets)):
        
        tofs = jd_arrival - jd_soi_current

        for c in np.flatnonzero(tofs > 0):

            tof_days = float(tofs[c])
            n_lambert += 1
   
            lambert = pk.lambert_problem(
                r0=departures_current[c],
                r1=r_target,
                tof=tof_days * day,
                mu=mu_sun,
                cw=False,
                multi_revs=0,
            )

            v_lambert = np.asarray(lambert.v0[0])

            distance_squared = np.sum((vinf_current[c] - v_lambert[:2])**2)
            compatible = distance_squared <= radius_v[c]**2

            if not compatible:
                continue

            rendezvous_dv.append(np.linalg.norm(arrival_velocities[j] - np.asarray(lambert.v1[0])))
            matches.append((float(tof_days), int(cluster_ids[c])))

            tof_days_grid.append(tof_days)
            lambert_velocities.append(v_lambert)
            lambert_compatible.append(compatible)
            lambert_cluster_rows.append(c)

            lambert_departures.append(departures_current[c].copy())
            lambert_arrivals.append(r_target.copy())
            lambert_etd_indices.append(i_etd)
            lambert_target_indices.append(j)


tof_days_grid = np.asarray(tof_days_grid)
lambert_velocities = np.asarray(lambert_velocities).reshape(-1, 3)
lambert_compatible = np.asarray(lambert_compatible, dtype=bool)
lambert_cluster_rows = np.asarray(lambert_cluster_rows, dtype=int)
lambert_departures = np.asarray(lambert_departures).reshape(-1, 3)
lambert_arrivals = np.asarray(lambert_arrivals).reshape(-1, 3)
lambert_etd_indices = np.asarray(lambert_etd_indices, dtype=int)
lambert_target_indices = np.asarray(lambert_target_indices, dtype=int)
rendezvous_dv = np.asarray(rendezvous_dv)


if PRINT:
    print("Lambert calcolati:", n_lambert)
    print("Coppie TOF-cluster compatibili:", len(matches))
    print("Cluster compatibili:", len({c for _, c in matches}))



elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.2f} s\n")



# Plot database and heliocentric targets
if PLOT or SAVE:

    from matplotlib.ticker import MaxNLocator

    def plot_accepted_orbits(member_orbits, member_cluster_rows, dv_by_cluster, *,
                             title, dv_norm, dv_cmap, bins, total_orbits):
        """Plot catalogue members once, coloured by their cluster's minimum DV.

        Inputs contain one row per unique member, with (eps, e, omega_deg) in the
        common catalogue frame and its zero-based cluster row. Finite cluster DVs
        are accepted. Reuse fixed dv_norm and bins['eps'/'e'/'omega'] for all ETDs
        and the global figure. Identical elements can represent distinct members.
        total_orbits includes unassigned members for the grey catalogue background.
        Returns None if no members are accepted; otherwise returns the figure.
        """
        # One membership lookup: successful-transfer multiplicity never enters.
        member_dv = np.asarray(dv_by_cluster)[np.asarray(member_cluster_rows, dtype=int)]
        accepted = np.isfinite(member_dv)
        if not np.any(accepted):
            return None
        values = np.asarray(member_orbits, dtype=float)[accepted].copy()
        values[:, 2] %= 360.0
        total_values = np.asarray(total_orbits, dtype=float).copy()
        total_values[:, 2] %= 360.0
        colours = member_dv[accepted]
        n_clusters = np.unique(np.asarray(member_cluster_rows)[accepted]).size
        edges = [np.asarray(bins[key]) for key in ("eps", "e", "omega")]
        if any(np.any(values[:, i] < edge[0]) or np.any(values[:, i] > edge[-1])
               for i, edge in enumerate(edges)):
            raise ValueError("shared histogram bins must cover every accepted member")

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        labels = (r"$\epsilon$ [km$^2$/s$^2$]", r"$e$ [-]",
                  r"$\omega$ catalogo mod $360^\circ$")
        for ax, (x, y) in zip(axes[0], ((0, 1), (0, 2), (1, 2))):
            scatter = ax.scatter(
                values[:, x], values[:, y], c=colours, norm=dv_norm,
                cmap=dv_cmap, s=14, alpha=0.8, linewidths=0,
            )
            ax.set(xlabel=labels[x], ylabel=labels[y],
                   xlim=(edges[x][0], edges[x][-1]),
                   ylim=(edges[y][0], edges[y][-1]))
            ax.grid(alpha=0.2)
        colourbar = fig.colorbar(scatter, ax=axes[0].tolist(), pad=0.02)
        colourbar.set_label("ΔV rendez-vous min [km/s]")
        for column, ax in enumerate(axes[1]):
            ax.hist(total_values[:, column], bins=edges[column], color="0.7",
                    alpha=0.3, label="Totale catalogo", zorder=1)
            ax.hist(values[:, column], bins=edges[column], color="steelblue",
                    edgecolor="white", linewidth=0.5, label="Accettate", zorder=2)
            if column == 0:
                ax.legend(fontsize="small")
            ax.set(xlabel=labels[column], ylabel="Numero di traiettorie",
                   xlim=(edges[column][0], edges[column][-1]))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            f"{title}\n{len(values)} traiettorie / {n_clusters} cluster — "
            "orbite terrestri al perigeo, riferimento del catalogo\n"
            "min ΔV per cluster (modello medio)"
        )
        return fig


    successful_etds = np.unique(lambert_etd_indices[lambert_compatible])

    if successful_etds.size == 0:
        print("Nessun ETD compatibile: nessuna immagine da generare.")

    else:
        if SAVE:
            output_dir = (Path(__file__).resolve().parent / "GA18_plots" / f"asteroide_{TARGET_ID}_{time.strftime('%Y%m%d_%H%M%S')}")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Members are read only for the new orbital-distribution plots.
        member_file = Path(DATA_FILE).with_name("trajectory_exit_catalog.txt")
        member_catalog = np.loadtxt(member_file, ndmin=2)
        if member_catalog.shape[1] < 12:
            raise ValueError("Per i nuovi plot serve il catalogo GA16 esteso a 12 colonne.")
        members = member_catalog[member_catalog[:, 1] >= 0]
        member_ids, member_counts = np.unique(members[:, 1].astype(int), return_counts=True)
        if (not np.array_equal(member_ids, cluster_ids)
                or not np.array_equal(member_counts, cluster_sizes)
                or len(np.unique(members[:, 0])) != len(members)):
            raise ValueError("Catalogo dei membri e statistiche dei cluster non coerenti.")
        member_cluster_rows = np.searchsorted(cluster_ids, members[:, 1].astype(int))
        member_orbits = members[:, 9:12].copy()
        member_orbits[:, 2] %= 360.0

        # Minimum rendez-vous DV per cluster; no repeated members in the union.
        global_cluster_dv = np.full(len(cluster_ids), np.inf)
        np.minimum.at(global_cluster_dv, lambert_cluster_rows[lambert_compatible],
                      rendezvous_dv[lambert_compatible])
        global_member_mask = np.isfinite(global_cluster_dv[member_cluster_rows])
        orbital_bins = {
            "eps": np.histogram_bin_edges(member_catalog[:, 9], bins=30),
            "e": np.histogram_bin_edges(member_catalog[:, 10], bins=30),
            "omega": np.linspace(0.0, 360.0, 37),
        }
        dv_min = float(rendezvous_dv[lambert_compatible].min())
        dv_max = float(rendezvous_dv[lambert_compatible].max())
        if dv_min == dv_max:
            dv_max = dv_min + max(1e-6, abs(dv_min) * 0.01)
        dv_norm = plt.Normalize(dv_min, dv_max)
        dv_cmap = plt.get_cmap("viridis_r")

        if SAVE:
            np.savetxt(
                output_dir / "transfers_rendezvous.txt",
                np.column_stack((
                    jd_etd_grid[lambert_etd_indices[lambert_compatible]],
                    cluster_ids[lambert_cluster_rows[lambert_compatible]],
                    jd_targets[lambert_target_indices[lambert_compatible]],
                    tof_days_grid[lambert_compatible], rendezvous_dv[lambert_compatible],
                )),
                fmt=("%.17e", "%d", "%.17e", "%.17e", "%.17e"),
                header=("jd_etd_tdb cluster_id jd_arrival_tdb tof_from_soi_days dv_rendezvous_km_s\n"
                        "Arrival impulse only; departures use cluster mean states.\n"
                        f"Catalog: {DATA_FILE}"),
            )

        # Scala TOF comune a tutte le immagini
        tof_cmap = plt.get_cmap("viridis")
        tof_min = float(tof_days_grid[lambert_compatible].min())
        tof_max = float(tof_days_grid[lambert_compatible].max())

        if tof_min == tof_max:
            tof_min -= 0.5
            tof_max += 0.5

        tof_norm = plt.Normalize(tof_min, tof_max)

        # Geometria di sfondo, calcolata una sola volta
        orbit_points = np.array([pk.par2ic([*asteroid_elements[TARGET_ID][:5], nu], mu_sun)[0] for nu in np.linspace(0, 2 * np.pi, 500)])

        earth_orbit = earth_moon_geometry(np.linspace(JD_START, JD_END, 400))[0]

        angles = np.linspace(0, 2 * np.pi, 200)

        for i_etd in successful_etds:

            JD_ETD = float(jd_etd_grid[i_etd])

            # Tutte le soluzioni compatibili associate a questo ETD
            selected = (lambert_compatible & (lambert_etd_indices == i_etd))

            hit_indices = np.flatnonzero(selected)
            compatible_rows = np.unique(lambert_cluster_rows[selected])
            target_indices = np.unique(lambert_target_indices[selected])

            # Medie dei cluster nella geometria di questo ETD
            r_earth_etd, _, _, theta_moon_etd = earth_moon_geometry(JD_ETD)

            v_earth_soi = earth_moon_geometry(JD_ETD + t_soi)[1]

            mean_v_heliocentric = escape_velocity_to_heliocentric(
                mean_v_soi, v_earth_soi, theta_moon_etd
            )

            fig, axes = plt.subplots(1, 2, figsize=(16, 9), layout="constrained")

            fig.suptitle(
                f"Asteroide {TARGET_ID} — ETD JD {JD_ETD:.6f} TDB\n"
                f"{hit_indices.size} trasferimenti — "
                f"{target_indices.size} epoche di arrivo compatibili"
            )

            # PANNELLO SINISTRO: velocità e cluster
            ax = axes[0]

            for c, (center, radius) in enumerate(
                zip(mean_v_heliocentric, radius_v)
            ):
                active = c in compatible_rows

                ax.add_patch(plt.Circle(
                    center, radius,
                    fill=False,
                    edgecolor="tab:red" if active else "0.75",
                    linewidth=1.5 if active else 0.8,
                    alpha=1.0 if active else 0.7,
                ))

                if active:
                    ax.annotate(
                        str(cluster_ids[c]),
                        (center[0], center[1] + radius),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        color="darkred",
                        fontsize=9,
                    )

            ax.scatter(
                mean_v_heliocentric[:, 0],
                mean_v_heliocentric[:, 1],
                color="0.35", marker="x", s=20,
                label="Medie dei cluster",
            )

            ax.scatter(
                lambert_velocities[selected, 0],
                lambert_velocities[selected, 1],
                c=tof_days_grid[selected],
                cmap=tof_cmap, norm=tof_norm,
                s=30, edgecolors="black", linewidths=0.4,
                label="Soluzioni compatibili", zorder=5,
            )

            ax.set(
                xlabel="vx eliocentrica [km/s]",
                ylabel="vy eliocentrica [km/s]",
                title="Regioni di compatibilità",
            )
            ax.set_box_aspect(1)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper left", fontsize="small")

            # PANNELLO DESTRO: tutti i trasferimenti di questo ETD
            ax = axes[1]

            ax.plot(
                orbit_points[:, 0], orbit_points[:, 1],
                color="tab:purple", linewidth=1.5,
                label="Orbita asteroide",
            )

            ax.plot(
                earth_orbit[:, 0], earth_orbit[:, 1],
                color="tab:red", linewidth=1,
                label="Traiettoria Terra",
            )

            for radius_au in (R_MIN_AU, R_MAX_AU):
                ax.plot(
                    radius_au * d_E * np.cos(angles),
                    radius_au * d_E * np.sin(angles),
                    "k--", linewidth=1,
                )

            ax.plot(0, 0, "o", color="tab:orange", label="Sole")

            ax.plot(
                r_earth_etd[0], r_earth_etd[1],
                "o", color="tab:red", markersize=4,
                label="Terra a ETD",
            )

            ax.scatter(
                lambert_departures[selected, 0],
                lambert_departures[selected, 1],
                s=12, color="tab:green",
                label="Partenze alla SOI", zorder=4,
            )

            # Propaga ciascun arco dalla propria partenza salvata
            extent = R_MAX_AU * d_E

            for idx in hit_indices:
                states = pk.propagate_lagrangian_grid(
                    rv=[
                        lambert_departures[idx],
                        lambert_velocities[idx],
                    ],
                    tofs=np.linspace(0, tof_days_grid[idx] * day, 250),
                    mu=mu_sun,
                )

                positions = np.asarray(states)[:, 0, :]

                extent = max(extent, float(np.abs(positions[:, :2]).max()))

                ax.plot(
                    positions[:, 0], positions[:, 1],
                    color=tof_cmap(tof_norm(tof_days_grid[idx])),
                    linewidth=1, alpha=0.75, zorder=3,
                )

            # Tutti i punti di arrivo associati a questo ETD
            ax.scatter(
                lambert_arrivals[selected, 0],
                lambert_arrivals[selected, 1],
                c=tof_days_grid[selected],
                cmap=tof_cmap, norm=tof_norm,
                marker="*", s=85,
                edgecolors="black", linewidths=0.5,
                label="Arrivi compatibili", zorder=6,
            )

            ax.set(
                xlabel="x [km]",
                ylabel="y [km]",
                title="Trasferimenti e punti di arrivo",
                xlim=(-1.1 * extent, 1.1 * extent),
                ylim=(-1.1 * extent, 1.1 * extent),
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)
            ax.legend(fontsize="small")

            fig.colorbar(
                plt.cm.ScalarMappable(norm=tof_norm, cmap=tof_cmap),
                ax=axes.tolist(),
                label="TOF dalla SOI [giorni]",
                shrink=0.85,
            )

            if SAVE:
                filename = output_dir / (f"ETD_{i_etd:05d}_JD_{JD_ETD:.8f}.png")
                fig.savefig(filename, dpi=180)
                plt.close(fig)
            else:
                plt.show()

            cluster_dv = np.full(len(cluster_ids), np.inf)
            np.minimum.at(cluster_dv, lambert_cluster_rows[selected], rendezvous_dv[selected])
            orbital_fig = plot_accepted_orbits(
                member_orbits, member_cluster_rows, cluster_dv,
                title=f"Asteroide {TARGET_ID} — ETD JD {JD_ETD:.6f} TDB",
                dv_norm=dv_norm, dv_cmap=dv_cmap, bins=orbital_bins,
                total_orbits=member_catalog[:, 9:12],
            )
            if SAVE:
                orbital_fig.savefig(output_dir / f"orbite_ETD_{i_etd:05d}_JD_{JD_ETD:.8f}.png", dpi=180)
                plt.close(orbital_fig)
            else:
                plt.show()

        final_orbital_fig = plot_accepted_orbits(
            member_orbits, member_cluster_rows, global_cluster_dv,
            title=f"Asteroide {TARGET_ID} — tutte le ETD compatibili",
            dv_norm=dv_norm, dv_cmap=dv_cmap, bins=orbital_bins,
            total_orbits=member_catalog[:, 9:12],
        )
        if SAVE:
            final_orbital_fig.savefig(output_dir / "orbite_tutte_le_ETD.png", dpi=180)
            plt.close(final_orbital_fig)
            print(f"Salvate {2 * successful_etds.size + 1} immagini e transfers_rendezvous.txt in: {output_dir}\n")
        else:
            plt.show()
