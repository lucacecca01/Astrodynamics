from Celestial_Mechanics import Integration, Transformations
import matplotlib
matplotlib.use("Agg")
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
import gc
import json
from joblib import Parallel, delayed, parallel_config


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
mu_sun = pk.MU_SUN / 1e9       # km^3/s^2
AU_KM = pk.AU / 1000           # km
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

    rotated = []

    for velocity, theta in zip(velocities_catalog, theta_moon):

        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])

        rotated.append(rotation @ velocity)

    return np.array(rotated) + np.asarray(v_earth)[..., :2]





# Define parameters
N_SIGMA = 2.5                      # Number of standard deviations for the interval
N_TARGETS = 10000
SEED = 42
R_MIN_AU = 0.6
R_MAX_AU = 1.4





# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/cluster_velocity_statistics.txt"
EXCLUDED_CLUSTER_IDS = np.arange(763, 780)

data = np.loadtxt(DATA_FILE, ndmin=2)
data = data[~np.isin(data[:, 0].astype(int), EXCLUDED_CLUSTER_IDS)]

spice.furnsh(pk.udpla.de440s.kernel_file())

cluster_ids = data[:, 0].astype(int)
cluster_sizes = data[:, 1].astype(int)

mean_v_soi = data[:, 2:4]
std_v_soi = data[:, 4:6] 

t_etd = -data[:, 7]
std_t_etd = data[:, 9]

t_perigee = data[:, 6] + t_etd
std_t_perigee = data[:, 8] + std_t_etd

mean_r_soi = data[:, 10:12]
std_r_soi = data[:, 12:14]

direction = data[:, 14]



radius_v = N_SIGMA * np.linalg.norm(std_v_soi, axis=1)

if PRINT:
    print(f"\nCluster caricati: {len(cluster_ids)}")
    print(f"Traiettorie rappresentate: {cluster_sizes.sum()}")
    print(f"Limite radiale: {N_SIGMA:g} sigma")





# SOI exit epochs and heliocentric times of flight, in JD TDB and days.
START_WINDOW = 2461351.0
END_WINDOW = START_WINDOW + 1
STEP_SOI_DAYS = 1

MIN_TOF_DAYS = 350
MAX_TOF_DAYS = 1100
STEP_TOF_DAYS = 1

jd_soi_grid = START_WINDOW + np.arange(0, END_WINDOW - START_WINDOW, STEP_SOI_DAYS)
tof_search_days = np.arange(MIN_TOF_DAYS, MAX_TOF_DAYS, STEP_TOF_DAYS)

matches = []
tof_days_grid = []
lambert_velocities_departure = []
lambert_velocities_arrival = []
lambert_compatible = []
lambert_cluster_rows = []
lambert_capture_rows = []
lambert_branch_indices = []

lambert_departures = []
lambert_arrivals = []
lambert_soi_indices = []
lambert_tof_indices = []
rendezvous_dv = []

n_lambert = 0


def solve_soi(i_soi, jd_soi):

    out = [[] for _ in range(13)]

    (   matches, tof_days_grid,
        lambert_velocities_departure, lambert_velocities_arrival,
        lambert_compatible, lambert_cluster_rows,
        lambert_capture_rows, lambert_branch_indices,
        lambert_departures, lambert_arrivals,
        lambert_soi_indices, lambert_tof_indices,
        rendezvous_dv,
    ) = out

    n_lambert = 0

    r_earth, v_earth, _, theta_moon = earth_moon_geometry(jd_soi)

    departures = []

    phase_moon_etd = earth_moon_geometry(jd_soi + t_etd)[3]

    for position, total_angle in zip(mean_r_soi, phase_moon_etd):

        rotation = np.array([[np.cos(total_angle), -np.sin(total_angle)],
                             [np.sin(total_angle),  np.cos(total_angle)]])

        departure = r_earth.copy()
        departure[:2] += rotation @ position
        departures.append(departure)

    departures = np.array(departures)
    vinf = escape_velocity_to_heliocentric(mean_v_soi, v_earth, phase_moon_etd)

    mean_r_capture = mean_r_soi * np.array([1.0, -1.0])
    mean_v_capture = mean_v_soi * np.array([-1.0, 1.0])


    for j, tof in enumerate(tof_search_days):

        jd_arrival = jd_soi + tof
        r_target, v_target, _, theta_moon = earth_moon_geometry(jd_arrival)

        captures = []

        phase_moon_etd_capture = earth_moon_geometry(jd_arrival - t_etd)[3]

        for position, total_angle in zip(mean_r_capture, phase_moon_etd_capture):

            rotation = np.array([[np.cos(total_angle), -np.sin(total_angle)],
                                     [np.sin(total_angle),  np.cos(total_angle)]])

            capture = r_target.copy()
            capture[:2] += rotation @ position
            captures.append(capture)

        captures = np.array(captures)
        vinf_capture = escape_velocity_to_heliocentric(mean_v_capture, v_target, phase_moon_etd_capture)

        for c in range(len(cluster_ids)):
            for s in range(len(cluster_ids)):

                n_lambert += 1

                lambert = pk.lambert_problem(
                    r0=departures[c],
                    r1=captures[s],
                    tof=tof * day,
                    mu=mu_sun,
                    cw=False,
                    multi_revs=2,
                )

                for branch_index, (v0, v1) in enumerate(zip(lambert.v0, lambert.v1)):
                    v_lambert_departure = np.asarray(v0)
                    v_lambert_arrival = np.asarray(v1)

                    dx_0 = vinf[c, 0] - v_lambert_departure[0]
                    dy_0 = vinf[c, 1] - v_lambert_departure[1]

                    dx_1 = vinf_capture[s, 0] - v_lambert_arrival[0]
                    dy_1 = vinf_capture[s, 1] - v_lambert_arrival[1]

                    distance_squared_0 = dx_0 * dx_0 + dy_0 * dy_0
                    distance_squared_1 = dx_1 * dx_1 + dy_1 * dy_1

                    compatible = distance_squared_0 <= radius_v[c]**2 and distance_squared_1 <= radius_v[s]**2

                    if not compatible:
                        continue


                    t_check = np.linspace(0, tof * day, int(np.ceil(tof / 0.25)) + 1)
                    r_check = np.asarray(pk.propagate_lagrangian_grid(rv=[departures[c], v_lambert_departure], tofs=t_check, mu=mu_sun))[:, 0, :]
                    r_earth_check = earth_moon_geometry(jd_soi + t_check / day)[0]

                    fuori = np.linalg.norm(r_check - r_earth_check, axis=1) > E_SOI
                    indici_fuori = np.flatnonzero(fuori)

                    if indici_fuori.size == 0:
                        continue

                    if not np.all(fuori[indici_fuori[0]:indici_fuori[-1] + 1]):
                        min_distance = np.min(np.linalg.norm(r_check - r_earth_check, axis=1)[indici_fuori[0]:indici_fuori[-1] + 1])
                        print(f"Lambert {n_lambert}: incontro con la terra a d = {min_distance:.2f} km per {cluster_ids[c]} → {cluster_ids[s]} a JD {jd_soi:.6f} TDB, TOF {tof:.2f} giorni")
                        continue



                    rendezvous_dv.append(np.linalg.norm(v_target - v_lambert_arrival))
                    matches.append((float(tof), int(cluster_ids[c])))

                    tof_days_grid.append(tof)
                    lambert_velocities_departure.append(v_lambert_departure)
                    lambert_velocities_arrival.append(v_lambert_arrival)
                    lambert_compatible.append(compatible)
                    lambert_cluster_rows.append(c)
                    lambert_capture_rows.append(s)
                    lambert_branch_indices.append(branch_index)

                    lambert_departures.append(departures[c].copy())
                    lambert_arrivals.append(captures[s].copy())
                    lambert_soi_indices.append(i_soi)
                    lambert_tof_indices.append(j)

    return n_lambert, out



collected = (
    matches, tof_days_grid,
    lambert_velocities_departure, lambert_velocities_arrival,
    lambert_compatible, lambert_cluster_rows,
    lambert_capture_rows, lambert_branch_indices,
    lambert_departures, lambert_arrivals,
    lambert_soi_indices, lambert_tof_indices,
    rendezvous_dv,
)



with parallel_config(
    backend="loky",
    n_jobs=-1,
    inner_max_num_threads=1,
    initializer=spice.furnsh,
    initargs=(pk.udpla.de440s.kernel_file(),),
):
    results = Parallel(
        return_as="generator",
        batch_size=1,
        pre_dispatch="n_jobs",
    )(
        delayed(solve_soi)(i_soi, jd_soi)
        for i_soi, jd_soi in enumerate(jd_soi_grid)
    )

    for count, parts in results:
        n_lambert += count
        for destination, part in zip(collected, parts):
            destination.extend(part)



tof_days_grid = np.asarray(tof_days_grid)
lambert_velocities_departure = np.asarray(lambert_velocities_departure).reshape(-1, 3)
lambert_velocities_arrival = np.asarray(lambert_velocities_arrival).reshape(-1, 3)
lambert_compatible = np.asarray(lambert_compatible, dtype=bool)
lambert_cluster_rows = np.asarray(lambert_cluster_rows, dtype=int)
lambert_capture_rows = np.asarray(lambert_capture_rows, dtype=int)
lambert_branch_indices = np.asarray(lambert_branch_indices, dtype=int)
lambert_revolutions = (lambert_branch_indices + 1) // 2
lambert_departures = np.asarray(lambert_departures).reshape(-1, 3)
lambert_arrivals = np.asarray(lambert_arrivals).reshape(-1, 3)
lambert_soi_indices = np.asarray(lambert_soi_indices, dtype=int)
lambert_tof_indices = np.asarray(lambert_tof_indices, dtype=int)
lambert_arrival_epochs = jd_soi_grid[lambert_soi_indices] + tof_days_grid
rendezvous_dv = np.asarray(rendezvous_dv)


if PRINT:
    print("Lambert calcolati:", n_lambert)
    print("Coppie TOF-cluster compatibili:", len(matches))
    print("Cluster compatibili:", len({c for _, c in matches}))



elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.2f} s\n")




# Plot catalogue families compatible with Earth returns.
member_catalog = None
if PLOT or SAVE:

    from matplotlib.ticker import MaxNLocator

    def plot_accepted_orbits(member_orbits, member_cluster_rows, dv_by_cluster, *,
                            title, dv_norm, dv_cmap, bins, total_orbits=None,
                            incoming=False):
        """Plot members of candidate clusters, coloured by minimum SOI entry speed.

        Inputs contain one row per unique member, with (eps, e, omega_deg) in the
        common catalogue frame and its zero-based cluster row. Finite cluster DVs
        are accepted. Reuse fixed dv_norm and bins['eps'/'e'/'omega'] for all SOIs
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
        values[:, 2] = ((-1 if incoming else 1) * values[:, 2]) % 360.0

        if total_orbits is not None:
            total_values = np.asarray(total_orbits, dtype=float).copy()
            total_values[:, 2] = ((-1 if incoming else 1) * total_values[:, 2]) % 360.0

        colours = member_dv[accepted]
        n_clusters = np.unique(np.asarray(member_cluster_rows)[accepted]).size
        edges = [np.asarray(bins[key]) for key in ("eps", "e", "omega")]

        if any(np.any(values[:, i] < edge[0]) or np.any(values[:, i] > edge[-1])
            for i, edge in enumerate(edges)):
            raise ValueError("shared histogram bins must cover every accepted member")

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        labels = (r"$\epsilon$ [km$^2$/s$^2$]", r"$e$ [-]",
                  r"$\omega$ ingresso riflesso mod $360^\circ$" if incoming
                  else r"$\omega$ catalogo mod $360^\circ$")
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
        colourbar.set_label("Min velocità geocentrica all'ingresso SOI [km/s]")
        for column, ax in enumerate(axes[1]):
            if total_orbits is not None:
                ax.hist(total_values[:, column], bins=edges[column], color="0.7",
                        alpha=0.3, label="Totale catalogo", zorder=1)

            ax.hist(values[:, column], bins=edges[column], color="steelblue",
                    edgecolor="white", linewidth=0.5, label="Membri di cluster candidati", zorder=2)
            if column == 0:
                ax.legend(fontsize="small")
            ax.set(xlabel=labels[column], ylabel="Numero di traiettorie",
                xlim=(edges[column][0], edges[column][-1]))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            f"{title}\n{len(values)} membri / {n_clusters} cluster candidati — "
            "orbite terrestri al perigeo, riferimento del catalogo\n"
            "min velocità ingresso SOI per cluster (modello medio; membri non verificati singolarmente)"
        )
        return fig


    def plot_velocity_regions(ax, centres, velocities, active_rows, tofs, *,
                              title, xlabel, ylabel, norm, cmap):
        """Compare velocities and disks expressed in the same reference axes."""
        for row, (centre, radius) in enumerate(zip(centres, radius_v)):
            active = row in active_rows
            ax.add_patch(plt.Circle(
                centre, radius, fill=False,
                edgecolor="tab:red" if active else "0.75",
                linewidth=1.5 if active else 0.8,
                alpha=1.0 if active else 0.7,
            ))
            if active:
                ax.annotate(str(cluster_ids[row]), (centre[0], centre[1] + radius),
                            xytext=(0, 4), textcoords="offset points", ha="center",
                            color="darkred", fontsize=8)
        ax.scatter(centres[:, 0], centres[:, 1], color="0.35", marker="x", s=20,
                   label="Medie dei cluster")
        ax.scatter(velocities[:, 0], velocities[:, 1], c=tofs, cmap=cmap, norm=norm,
                   s=30, edgecolors="black", linewidths=0.4,
                   label="Soluzioni compatibili", zorder=5)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize="small")



    successful_sois = np.unique(lambert_soi_indices[lambert_compatible])

    if successful_sois.size == 0:
        print("Nessun SOI compatibile: nessuna immagine da generare.")

    else:
        saved_images = 0
        if SAVE:
            output_dir = Path(__file__).resolve().parent / "GA19_earth_returns_3"
            output_dir.mkdir(parents=True, exist_ok=True)

        if member_catalog is None:
            member_file = Path(DATA_FILE).with_name("trajectory_exit_catalog.txt")
            member_catalog = np.loadtxt(member_file, ndmin=2)
            member_catalog = member_catalog[~np.isin(member_catalog[:, 1].astype(int), EXCLUDED_CLUSTER_IDS)]

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
                    jd_soi_grid[lambert_soi_indices[lambert_compatible]],
                    cluster_ids[lambert_cluster_rows[lambert_compatible]],
                    cluster_ids[lambert_capture_rows[lambert_compatible]],
                    lambert_arrival_epochs[lambert_compatible],
                    tof_days_grid[lambert_compatible], rendezvous_dv[lambert_compatible],
                    lambert_branch_indices[lambert_compatible],
                    lambert_revolutions[lambert_compatible],
                )),
                fmt=("%.17e", "%d", "%d", "%.17e", "%.17e", "%.17e", "%d", "%d"),
                header=("jd_soi_tdb departure_cluster_id capture_cluster_id jd_arrival_tdb "
                        "tof_from_soi_days arrival_geocentric_speed_km_s lambert_branch_index revolutions\n"
                        "Compatibility of outgoing/incoming cluster mean states; members not individually verified.\n"
                        "Arrival speed is relative to Earth at SOI entry, not a capture manoeuvre cost.\n"
                        "Target: reflected time-reversed SOI family; planar ECLIPJ2000 ephemerides.\n"
                        "jd_arrival_tdb = jd_soi_tdb + tof_from_soi_days.\n"
                        f"SOI window [JD TDB, end excluded]: {START_WINDOW} {END_WINDOW}; "
                        f"step [days]: {STEP_SOI_DAYS}\n"
                        f"TOF range [days, max excluded]: {MIN_TOF_DAYS} {MAX_TOF_DAYS}; "
                        f"step [days]: {STEP_TOF_DAYS}\n"
                        f"Catalog: {DATA_FILE}"
                    ),
            )

        if True:

            # Scala TOF comune a tutte le immagini
            tof_cmap = plt.get_cmap("viridis")
            tof_min = float(tof_days_grid[lambert_compatible].min())
            tof_max = float(tof_days_grid[lambert_compatible].max())

            if tof_min == tof_max:
                tof_min -= 0.5
                tof_max += 0.5

            tof_norm = plt.Normalize(tof_min, tof_max)

            # Earth is both departure body and target; cover all plotted epochs.
            earth_orbit_epochs = np.linspace(
                jd_soi_grid[successful_sois].min(),
                lambert_arrival_epochs[lambert_compatible].max(),
                400,
            )
            earth_orbit = earth_moon_geometry(earth_orbit_epochs)[0]

            angles = np.linspace(0, 2 * np.pi, 200)

            # Each pair of nonzero-revolution Lambert branches belongs to one
            # revolution count. Keep identical colour scales for all groups.
            plot_groups = sorted(set(zip(
                lambert_soi_indices[lambert_compatible].tolist(),
                lambert_revolutions[lambert_compatible].tolist(),
            )))
            for i_soi, n_revolutions in plot_groups:

                JD_SOI = float(jd_soi_grid[i_soi])
                if SAVE:
                    revolution_output_dir = output_dir / f"rivoluzioni_{n_revolutions}"
                    revolution_output_dir.mkdir(parents=True, exist_ok=True)

                selected = (lambert_compatible & (lambert_soi_indices == i_soi)
                            & (lambert_revolutions == n_revolutions))

                hit_indices = np.flatnonzero(selected)
                compatible_rows = np.unique(lambert_cluster_rows[selected])
                capture_rows = np.unique(lambert_capture_rows[selected])
                arrival_epochs = np.unique(lambert_arrival_epochs[selected])

                # Medie dei cluster nella geometria di questo SOI
                r_earth_soi, v_earth_soi, _, theta_moon_soi = earth_moon_geometry(JD_SOI)
                theta_vinf_soi = earth_moon_geometry(JD_SOI + t_etd)[3]

                mean_v_heliocentric = escape_velocity_to_heliocentric(
                    mean_v_soi, v_earth_soi, theta_vinf_soi
                )

                # Arrival epochs differ: rotate each geocentric arrival velocity
                # into its own future-ETD axes before comparing with the same disks.
                v_earth_arrival = earth_moon_geometry(lambert_arrival_epochs[selected])[1]
                phase_capture = earth_moon_geometry(
                    lambert_arrival_epochs[selected] - t_etd[lambert_capture_rows[selected]]
                )[3]
                relative_arrival = (lambert_velocities_arrival[selected] - v_earth_arrival)[:, :2]
                cos_phase, sin_phase = np.cos(phase_capture), np.sin(phase_capture)
                arrival_in_etd_axes = np.column_stack((
                    cos_phase * relative_arrival[:, 0] + sin_phase * relative_arrival[:, 1],
                    -sin_phase * relative_arrival[:, 0] + cos_phase * relative_arrival[:, 1],
                ))
                mean_capture_in_etd_axes = mean_v_soi * np.array([-1.0, 1.0])

                fig, axes = plt.subplots(1, 3, figsize=(22, 8), layout="constrained")

                fig.suptitle(
                    f"Collegamenti SOI → SOI — {n_revolutions} rivoluzioni complete — "
                    f"uscita JD {JD_SOI:.6f} TDB\n"
                    f"{hit_indices.size} trasferimenti — {compatible_rows.size} cluster in uscita / "
                    f"{capture_rows.size} in ingresso — "
                    f"{arrival_epochs.size} epoche di arrivo compatibili"
                )

                # PANNELLO SINISTRO: velocità e cluster
                plot_velocity_regions(
                    axes[0], mean_v_heliocentric, lambert_velocities_departure[selected],
                    compatible_rows, tof_days_grid[selected],
                    title="Compatibilità uscita (eliocentrico)",
                    xlabel="vx eliocentrica [km/s]", ylabel="vy eliocentrica [km/s]",
                    norm=tof_norm, cmap=tof_cmap,
                )

                # PANNELLO CENTRALE: famiglie entranti nei rispettivi assi ETD.
                plot_velocity_regions(
                    axes[1], mean_capture_in_etd_axes, arrival_in_etd_axes, capture_rows,
                    tof_days_grid[selected], title="Compatibilità ingresso (assi ETD)",
                    xlabel="vx geocentrica [km/s]", ylabel="vy geocentrica [km/s]",
                    norm=tof_norm, cmap=tof_cmap,
                )

                # PANNELLO DESTRO: tutti i trasferimenti di questo SOI.
                ax = axes[2]

                ax.plot(
                    earth_orbit[:, 0], earth_orbit[:, 1],
                    color="tab:red", linewidth=1,
                    label="Traiettoria Terra",
                )

                for radius_au in (R_MIN_AU, R_MAX_AU):
                    ax.plot(
                        radius_au * AU_KM * np.cos(angles),
                        radius_au * AU_KM * np.sin(angles),
                        "k--", linewidth=1,
                    )

                ax.plot(0, 0, "o", color="tab:orange", label="Sole")

                ax.plot(
                    r_earth_soi[0], r_earth_soi[1],
                    "o", color="tab:red", markersize=4,
                    label="Terra all'uscita dalla SOI",
                )

                ax.scatter(
                    lambert_departures[selected, 0],
                    lambert_departures[selected, 1],
                    s=12, color="tab:green",
                    label="Partenze alla SOI", zorder=4,
                )

                # Propaga ciascun arco dalla propria partenza salvata
                extent = max(
                    R_MAX_AU * AU_KM,
                    float(np.abs(earth_orbit[:, :2]).max()),
                    float(np.abs(lambert_departures[selected, :2]).max()),
                    float(np.abs(lambert_arrivals[selected, :2]).max()),
                )

                for idx in hit_indices:
                    states = pk.propagate_lagrangian_grid(
                        rv=[
                            lambert_departures[idx],
                            lambert_velocities_departure[idx],
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

                # Tutti i punti di arrivo associati a questo SOI
                ax.scatter(
                    lambert_arrivals[selected, 0],
                    lambert_arrivals[selected, 1],
                    c=tof_days_grid[selected],
                    cmap=tof_cmap, norm=tof_norm,
                    marker="*", s=85,
                    edgecolors="black", linewidths=0.5,
                    label="Ingressi alla SOI", zorder=6,
                )

                ax.set(
                    xlabel="x [km]",
                    ylabel="y [km]",
                    title="Trasferimenti eliocentrici SOI → SOI",
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
                    filename = revolution_output_dir / (f"SOI_{i_soi:05d}_JD_{JD_SOI:.8f}.png")
                    fig.savefig(filename, dpi=180)
                    saved_images += 1
                    fig.clear()
                    plt.close(fig)
                    gc.collect()
                else:
                    plt.show()
                    plt.close(fig)

                cluster_dv = np.full(len(cluster_ids), np.inf)
                np.minimum.at(cluster_dv, lambert_cluster_rows[selected], rendezvous_dv[selected])
                orbital_fig = plot_accepted_orbits(
                    member_orbits, member_cluster_rows, cluster_dv,
                    title=f"Famiglie in uscita — {n_revolutions} rivoluzioni — SOI JD {JD_SOI:.6f} TDB",
                    dv_norm=dv_norm, dv_cmap=dv_cmap, bins=orbital_bins,
                )
                if SAVE:
                    orbital_fig.savefig(revolution_output_dir / f"orbite_uscita_SOI_{i_soi:05d}_JD_{JD_SOI:.8f}.png", dpi=180)
                    saved_images += 1
                    orbital_fig.clear()
                    plt.close(orbital_fig)
                    gc.collect()
                else:
                    plt.show()
                    plt.close(orbital_fig)

                capture_dv = np.full(len(cluster_ids), np.inf)
                np.minimum.at(capture_dv, lambert_capture_rows[selected], rendezvous_dv[selected])
                capture_fig = plot_accepted_orbits(
                    member_orbits, member_cluster_rows, capture_dv,
                    title=f"Famiglie in ingresso — {n_revolutions} rivoluzioni — uscita SOI JD {JD_SOI:.6f} TDB",
                    dv_norm=dv_norm, dv_cmap=dv_cmap, bins=orbital_bins, incoming=True,
                )
                if SAVE:
                    capture_fig.savefig(revolution_output_dir / f"orbite_ingresso_SOI_{i_soi:05d}_JD_{JD_SOI:.8f}.png", dpi=180)
                    saved_images += 1
                    capture_fig.clear()
                    plt.close(capture_fig)
                    gc.collect()
                else:
                    plt.show()
                    plt.close(capture_fig)

        for n_revolutions in np.unique(lambert_revolutions[lambert_compatible]):
            selected = lambert_compatible & (lambert_revolutions == n_revolutions)
            if SAVE:
                revolution_output_dir = output_dir / f"rivoluzioni_{n_revolutions}"
                revolution_output_dir.mkdir(parents=True, exist_ok=True)
            for incoming, role, rows in (
                (False, "uscita", lambert_cluster_rows),
                (True, "ingresso", lambert_capture_rows),
            ):
                cluster_speeds = np.full(len(cluster_ids), np.inf)
                np.minimum.at(cluster_speeds, rows[selected], rendezvous_dv[selected])
                final_orbital_fig = plot_accepted_orbits(
                    member_orbits, member_cluster_rows, cluster_speeds,
                    title=f"Famiglie in {role} — {n_revolutions} rivoluzioni — tutte le SOI compatibili",
                    dv_norm=dv_norm, dv_cmap=dv_cmap, bins=orbital_bins,
                    total_orbits=member_catalog[:, 9:12], incoming=incoming,
                )
                if SAVE:
                    final_orbital_fig.savefig(revolution_output_dir / f"orbite_{role}_tutte_le_SOI.png", dpi=180)
                    saved_images += 1
                    final_orbital_fig.clear()
                    plt.close(final_orbital_fig)
                    gc.collect()
                else:
                    plt.show()
                    plt.close(final_orbital_fig)

        if SAVE:
            print(f"Salvate {saved_images} immagini e transfers_rendezvous.txt in: {output_dir}\n")


print("Scansione dei ritorni sulla Terra completata.\n")
