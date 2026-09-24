from Celestial_Mechanics import Integration, Transformations
from pathlib import Path
from multiprocessing import get_context
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


SAVE = True
PLOT_CLUSTERS = False
PLOT_MEDOIDS = False



if not PLOT_CLUSTERS:
    plt.ioff()



# Save figure function
def save_figure(fig, name):

    if SAVE:
        fig.savefig(output_directory / name, dpi=200)
    if not PLOT_CLUSTERS:
        plt.close(fig)



#Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)

    return E_SOI / d - r_earth



# Define collision events
def collision(t, x, mu):

    r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
    r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Define perigee events
def earth_perigee(t, x, mu):

    return (x[0] + mu)*x[3] + x[1]*x[4] + x[2]*x[5]



# Compute geocentric osculating parameters
def orbital_parameters(t, x):

    x_earth = np.asarray(x, dtype=float).copy()
    x_earth[0] += mu
    x_inertial = Transformations.CR3BP_to_inertial(x_earth, 1, t)[:, 0]

    r = x_inertial[:3] * d
    v = x_inertial[3:] * d / TU

    with np.errstate(divide="ignore", invalid="ignore"):
        _, a, e, _, _, w, _ = Transformations.coe_from_sv(r, v, mu_earth)

    eps = 0.5 * np.dot(v, v) - mu_earth / np.linalg.norm(r)
    parameters = np.array([eps, e, w])
    parameters[~np.isfinite(parameters)] = np.nan

    return parameters



# Select the first eligible perigee
def first_perigee_parameters(sol):

    collision_time = (abs(sol.t_events[1][0] - T_min) if len(sol.t_events[1]) else np.inf)

    if collision(T_min, sol.y[:, 0], mu) <= 0:
        return np.full(3, np.nan), np.nan

    for t, x in zip(sol.t_events[0], sol.y_events[0]):

        if abs(t - T_min) >= collision_time:
            break

        r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
        r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

        if abs(t - T_min) > 1e-10 and r_moon > 2 * M_SOI / d:
            return orbital_parameters(t, x), t

    return np.full(3, np.nan), np.nan



# Define integration function
def integrate_one(x0, T_min, T_max, dt, events=None, rtol=1e-9):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=rtol,
        events=events,
        dense_output=True
    )

    return sol



# Define integration and sampling function
def integrate_and_sample_one(x0):

    solution_b = integrate_one(x0, T_min, T_max_b, dt_b, (earth_perigee, collision))

    parameters_b, t_perigee = first_perigee_parameters(solution_b)

    if not np.isfinite(t_perigee):
        return None

    if not np.all(np.isfinite(parameters_b)):
        raise RuntimeError("Parametri al perigeo non finiti.")

    x_perigee = solution_b.sol(t_perigee)
    rx = x_perigee[0] + mu
    ry = x_perigee[1]

    hz = rx * (x_perigee[4] + rx) - ry * (x_perigee[3] - ry)
    direction = np.sign(hz)

    x_b = solution_b.sol(np.linspace(t_perigee, T_min, N_PLOT))[:2].copy()

    del solution_b


    solution_f = integrate_one(x0, T_min, T_max_f, dt_f, (earth_SOI_exit, collision))

    if not solution_f.success or len(solution_f.t_events[0]) == 0:
        raise RuntimeError("Integrazione forward fallita o SOI non raggiunta.")

    t_soi = solution_f.t_events[0][0]
    x_soi = solution_f.y_events[0][0].copy()

    if len(solution_f.t_events[1]) and solution_f.t_events[1][0] <= t_soi:
        raise RuntimeError("Collisione precedente all'uscita dalla SOI.")

    x_f = solution_f.sol(np.linspace(T_min, t_soi, N_PLOT)[1:])[:2].copy()

    x_soi[0] += mu
    x_inertial = Transformations.CR3BP_to_inertial(x_soi, 1, t_soi)[:, 0]
    velocity_f = x_inertial[3:] * d / TU
        
    del solution_f, 


    features = np.concatenate((parameters_b, velocity_f, [direction]))

    return features, np.hstack((x_b, x_f))



# Define normalization function for features
def normalize_block(features):

    centered = features - np.mean(features, axis=0)
    scale = np.linalg.norm(centered) / np.sqrt(len(centered))

    if scale <= 1e-14:
        raise ValueError("Feature block has zero variance.")

    return centered / scale



# Define farthest point selection function
def farthest_point_selection(features, max_representatives):

    feature_center = np.mean(features, axis=0)

    first_id = np.argmin(np.sum((features - feature_center)**2, axis=1))

    representative_ids = [first_id]

    minimum_distances_squared = np.sum((features - features[first_id])**2, axis=1)

    labels = np.zeros(len(features), dtype=int)

    radius_history = [np.sqrt(np.max(minimum_distances_squared))]

    for cluster_id in range(1, max_representatives):

        next_id = np.argmax(minimum_distances_squared)

        if minimum_distances_squared[next_id] <= 1e-14:
            break

        representative_ids.append(next_id)

        new_distances_squared = np.sum((features - features[next_id])**2, axis=1)

        closer = (new_distances_squared < minimum_distances_squared)

        minimum_distances_squared[closer] = (new_distances_squared[closer])

        labels[closer] = cluster_id

        radius_history.append(np.sqrt(np.max(minimum_distances_squared)))

    return (np.asarray(representative_ids), labels, np.asarray(radius_history), np.sqrt(minimum_distances_squared))




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
E_SOI = (m1 / ms)**(2/5) * d_E
M_SOI = (m2 / m1)**(2/5) * d
mu_earth = G * m1




# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP.txt"

database = np.atleast_2d(np.loadtxt(DATA_FILE, skiprows=1))

columns = {
    "Cj": 6,
    "E_SOI": 7,
    "h_min_moon": 8,
    "h_perigee": 9,
}

valid_mask = (database[:, columns["h_min_moon"]] >= 0) & (database[:, columns["h_perigee"]] >= 0)
valid_ids = np.where(valid_mask)[0]

source_ids = valid_ids[::1]
selection = database[source_ids, :]
x0_selection = selection[:, :6]


print(f"\nDatabase trajectories: {len(database)}")
print(f"Valid trajectories:    {len(valid_ids)}")
print(f"Selected trajectories: {len(x0_selection)}")


# Initialize the packages
start_time = time.perf_counter()
Integrator = Integration()



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0

T_max_f = 4 * np.pi
dt_f = 0.001

T_max_b = -4 * np.pi
dt_b = -0.001

event_features = []
N_PLOT = 500



# Define events
earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1

earth_perigee.terminal = False
earth_perigee.direction = -1

collision.terminal = False
collision.direction = 0




# Parallel integration backward and forward
event_features = np.empty((len(x0_selection), 7), dtype=np.float64)
X_curvature = np.empty((len(x0_selection), 2, 2 * N_PLOT - 1), dtype=np.float64)
keep = np.zeros(len(x0_selection), dtype=bool)
n_kept = 0

with get_context("fork").Pool() as pool:

    for i, result in enumerate(
        pool.imap(
            integrate_and_sample_one,
            x0_selection,
            chunksize=10,
        )
    ):
        if result is None:
            continue

        event_features[n_kept], X_curvature[n_kept] = result
        keep[i] = True
        n_kept += 1


if n_kept == 0:
    raise RuntimeError("Nessuna traiettoria con perigeo ammissibile.")

event_features = event_features[:n_kept]
X_curvature = X_curvature[:n_kept]
source_ids = source_ids[keep]
selection = database[source_ids, :]
x0_selection = selection[:, :6]

print(f"Traiettorie mantenute: {n_kept}; escluse: {len(keep) - n_kept}")



# Prepare clustering features
w = np.deg2rad(event_features[:, 2])
angular_features = np.column_stack((np.cos(w), np.sin(w)))

clustering_features = np.column_stack((
    normalize_block(event_features[:, 0:1]),   # eps
    normalize_block(event_features[:, 1:2]),   # e
    normalize_block(angular_features),         # w
    100 *normalize_block(event_features[:, 6:7]),                      # verso al perigeo
 #  normalize_block(event_features[:, 3:6]),   # v_SOI
))

#clustering_features[:, :4] /= np.sqrt(3)



# Perform farthest point selection
max_representatives = 33

representative_ids, labels, radius_history, minimum_distances = (
    farthest_point_selection(
        clustering_features,
        max_representatives,
    )
)

representative_source_ids = source_ids[representative_ids]

farthest_distances = minimum_distances.copy()
farthest_representative_ids = representative_ids.copy()

print(f"\nRepresentatives: {len(representative_ids)}")
print(f"Final mean distance: {np.mean(minimum_distances):.6f}")
print(f"Final 95th percentile: {np.percentile(minimum_distances, 95):.6f}")
print(f"Final 99th percentile: {np.percentile(minimum_distances, 99):.6f}")
print(f"Final maximum distance: {np.max(minimum_distances):.6f}")



# Define K values for KMeans clustering
k_values = np.unique(np.linspace(10, len(farthest_representative_ids), 10, dtype=int))

cluster_counts = []
mean_stds = []
small_cluster_counts = []
small_trajectory_fractions = []
escape_stats = []

output_directory = (Path(__file__).resolve().parent / f"Clusters/GA_14_plots_K{len(representative_ids)}")
output_directory.mkdir(parents=True, exist_ok=True)

physical_std_history = []
outlier_cluster_percent = []
outlier_cluster_percent_2sigma = []


# Refine farthest-point clusters with KMeans
for k in k_values:
    
    kmeans = KMeans(
        n_clusters=int(k),
        init=clustering_features[farthest_representative_ids[:k]],
        n_init=1,
        max_iter=500,
        tol=1e-9,
        algorithm="lloyd",
    )

    kmeans_labels = kmeans.fit_predict(clustering_features)

    representative_ids = np.empty(len(kmeans.cluster_centers_), dtype=int)

    for cluster_id, cluster_center in enumerate(kmeans.cluster_centers_):

        cluster_ids = np.flatnonzero(kmeans_labels == cluster_id)

        distances_squared = np.sum((clustering_features[cluster_ids] - cluster_center)**2, axis=1)

        representative_ids[cluster_id] = cluster_ids[np.argmin(distances_squared)]


    labels, minimum_distances = pairwise_distances_argmin_min(clustering_features, clustering_features[representative_ids])

    representative_source_ids = source_ids[representative_ids]

    cluster_ids, sizes = np.unique(labels, return_counts=True)

    variances = [np.mean(np.var(clustering_features[labels == c], axis=0)) for c in cluster_ids]

    mean_std = np.sqrt(np.average(variances, weights=sizes))

    variances = []
    physical_stds = []
    outlier_clusters = 0
    outlier_clusters_2sigma = 0

    for c in cluster_ids:
        mask = labels == c
        features_c = clustering_features[mask]

        variance_c = np.var(features_c, axis=0)
        variances.append(np.mean(variance_c))

        delta = features_c - np.mean(features_c, axis=0)
        distance_squared = np.sum(delta**2, axis=1)
        sigma_total_squared = np.sum(variance_c)

        outlier_clusters += int(np.any(distance_squared > 9 * sigma_total_squared))
        outlier_clusters_2sigma += int(np.any(distance_squared > 4 * sigma_total_squared))

        physical_c = event_features[mask, :5]

        angle_mean = np.mean(np.exp(1j * np.deg2rad(physical_c[:, 2])))

        if abs(angle_mean) < 1e-8:
            physical_c[:, 2] = np.nan
        else:
            origin = np.rad2deg(np.angle(angle_mean))
            physical_c[:, 2] = (physical_c[:, 2] - origin + 180) % 360 - 180

        physical_stds.append(np.std(physical_c, axis=0))

        if k == k_values[-1]:
            v_c = event_features[mask, 3:6]
            speed_c = np.linalg.norm(v_c, axis=1)
            valid = speed_c > 0

            direction_spread = np.nan
            if np.any(valid):
                unit_v = v_c[valid] / speed_c[valid, None]
                direction_spread = np.clip(1.0 - np.linalg.norm(np.mean(unit_v, axis=0)), 0, 1)

            escape_stats.append([np.std(speed_c), direction_spread])


    physical_stds = np.asarray(physical_stds)
    valid_counts = np.sum(np.isfinite(physical_stds), axis=0)

    physical_std_history.append(np.divide(
        np.nansum(physical_stds, axis=0),
        valid_counts,
        out=np.full(5, np.nan),
        where=valid_counts > 0,
    ))

    outlier_cluster_percent.append(100 * outlier_clusters / len(cluster_ids))
    outlier_cluster_percent_2sigma.append(100 * outlier_clusters_2sigma / len(cluster_ids))

    cluster_counts.append(len(cluster_ids))
    mean_stds.append(mean_std)

    print(f"K = {len(cluster_ids)}, STD = {mean_std:.6f}", flush=True)


    masks = (sizes == 1, sizes < 10, sizes < 100)

    small_cluster_counts.append([np.count_nonzero(mask) for mask in masks])
    small_trajectory_fractions.append([np.sum(sizes[mask]) / len(labels) for mask in masks])



# Plot mean within-cluster physical STD vs number of clusters
if SAVE or PLOT_CLUSTERS:
    history = np.asarray(physical_std_history)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    names = ["eps [km^2/s^2]", "e [-]", "w [deg]"]

    for j, name in enumerate(names):
        axes[j].plot(cluster_counts, history[:, j], "o-")
        axes[j].set_xlabel("Number of clusters")
        axes[j].set_ylabel(f"Mean STD — {name}")
        axes[j].grid(alpha=0.3)

        k_plot = np.asarray(cluster_counts)
    small_numbers = np.asarray(small_cluster_counts)

    axes[3].plot(k_plot, history[:, 3], "o-", label="vx")
    axes[3].plot(k_plot, history[:, 4], "o-", label="vy")
    axes[3].set_ylabel("Mean STD — v SOI [km/s]")

    axes[4].plot(k_plot, np.rint(np.asarray(outlier_cluster_percent_2sigma) * k_plot / 100), "o-", color="darkorange", label="At least one outlier > 2σ")
    axes[4].plot(k_plot, np.rint(np.asarray(outlier_cluster_percent) * k_plot / 100), "o-", color="darkred", label="At least one outlier > 3σ")
    axes[4].plot(k_plot, small_numbers[:, 1], "s--", color="blue", label="< 10 trajectories")
    axes[4].plot(k_plot, small_numbers[:, 2], "s--", color="green", label="< 100 trajectories")
    axes[4].set_ylabel("Number of clusters")
    axes[4].set_title("Initial-state outliers and cluster sizes")
    axes[4].set_ylim(bottom=0)

    for ax in axes[3:5]:
        ax.set_xlabel("Number of clusters")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    small_percent = (100 * np.asarray(small_cluster_counts) / np.asarray(cluster_counts)[:, None])

    axes[5].plot(cluster_counts, outlier_cluster_percent_2sigma, "o-", color="darkorange", label="At least one outlier > 2σ")
    axes[5].plot(cluster_counts, outlier_cluster_percent, "o-", color="darkred", label="At least one outlier > 3σ")
    axes[5].plot(cluster_counts, small_percent[:, 1], "s--", color="blue", label="< 10 trajectories")
    axes[5].plot(cluster_counts, small_percent[:, 2], "s--", color="green", label="< 100 trajectories")

    axes[5].set_xlabel("Number of clusters")
    axes[5].set_ylabel("Clusters [%]")
    axes[5].set_title("Initial-state outliers and cluster sizes")
    axes[5].set_ylim(0, 100)
    axes[5].grid(alpha=0.3)
    axes[5].legend(fontsize=8)

    save_figure(fig, "physical_STD_and_3sigma_vs_K.png")



    diagnostics = np.column_stack((physical_stds[:, :3], np.asarray(escape_stats), sizes))

    names = [
        "STD eps [km^2/s^2]",
        "STD e [-]",
        "STD w [deg]",
        "STD |v SOI| [km/s]",
        "Direction dispersion [0, 1]",
        "Number of trajectories",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    for j, ax in enumerate(axes.ravel()):
        ax.scatter(cluster_ids, diagnostics[:, j], s=15)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(names[j])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    axes.ravel()[4].set_ylim(0, 1)

    fig.suptitle(f"Initial compactness and escape diversity — K={len(cluster_ids)}")

    save_figure(fig, f"cluster_diagnostics_K{len(cluster_ids)}.png")    





# Plot cluster dispersion vs number of clusters
if SAVE:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(cluster_counts, mean_stds, "o-", color="blue")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Within-cluster STD")
    ax.set_title("Cluster dispersion vs number of clusters")

    fig.savefig(output_directory / "STD_vs_K.png", dpi=200)
    plt.close(fig)


print("\n=== KMEANS + MEDOIDS ===")
print(f"KMeans iterations: {kmeans.n_iter_}")
print(f"Representatives: {len(representative_ids)}")
print(f"Final mean distance: {np.mean(minimum_distances):.6f}")
print(f"Final 95th percentile: {np.percentile(minimum_distances, 95):.6f}")
print(f"Final 99th percentile: {np.percentile(minimum_distances, 99):.6f}")
print(f"Final maximum distance: {np.max(minimum_distances):.6f}")










# Compute cluster statistics
unique_clusters, cluster_sizes = np.unique(labels, return_counts=True)

cluster_standard_deviations = []
cluster_maximum_distances = []

n_clusters = len(unique_clusters)

cluster_variances = [np.mean(np.var(clustering_features[labels == cluster_id], axis=0))
    for cluster_id in unique_clusters]

mean_variance = np.average(cluster_variances, weights=cluster_sizes)


for cluster_id in unique_clusters:

    cluster_mask = labels == cluster_id
    cluster_features = clustering_features[cluster_mask]
    cluster_standard_deviations.append(np.sqrt(np.sum(np.var(cluster_features, axis=0))))

    cluster_maximum_distances.append(np.max(minimum_distances[cluster_mask]))


mean_cluster_standard_deviation = np.mean(cluster_standard_deviations)

mean_cluster_maximum_distance = np.mean(cluster_maximum_distances)



print(f"\nNumber of clusters: {n_clusters}")
print(f"Mean within-cluster standard deviation: {np.sqrt(mean_variance):.6f}")
print(f"\nMean cluster STD: {mean_cluster_standard_deviation:.6f}")
print(f"Mean cluster maximum distance: {mean_cluster_maximum_distance:.6f}\n")





# Mean within-cluster STD of physical quantities
velocity_unit = d / TU  

physical_quantities = {
    "Initial speed [km/s]"          :(np.linalg.norm(x0_selection[:, 3:6], axis=1) * velocity_unit),
    "Cj [-]"                        :selection[:, columns["Cj"]],
    "E_SOI [km^2/s^2]"              :selection[:, columns["E_SOI"]],
    "Minimum lunar altitude [km]"   :selection[:, columns["h_min_moon"]],
    "Perigee altitude [km]"         :selection[:, columns["h_perigee"]],
}

mean_physical_stds = {}

for quantity_name, values in physical_quantities.items():

    cluster_stds = [np.std(values[labels == cluster_id], ddof=1) for cluster_id in unique_clusters
        if np.sum(labels == cluster_id) > 1]

    mean_physical_stds[quantity_name] = np.mean(cluster_stds)


print("\n=== MEAN WITHIN-CLUSTER PHYSICAL STD ===\n" + "\n".join(f"{quantity_name}: {mean_std:.6f}" for quantity_name, mean_std in mean_physical_stds.items()) + "\n")







# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")




# Plot clusters and representative trajectories
if PLOT_CLUSTERS or SAVE:

    cmap = plt.get_cmap("turbo", max(n_clusters, 1),)

    cluster_colors = {cluster_id: cmap(color_id) for color_id, cluster_id in enumerate(unique_clusters)}

    groups = []

    for cluster_id in unique_clusters:
        groups.append((cluster_id, f"Cluster {cluster_id}"))


    max_panels = 24
    max_columns = 6


    for start in range(0, len(groups), max_panels):

        plot_groups = groups[start:start + max_panels]

        n_panels = len(plot_groups)

        ncols = min(max_columns, n_panels)

        nrows = int(np.ceil(n_panels / ncols))


        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9), constrained_layout=True, squeeze=False)

        axes = axes.ravel()


        for ax, (group_id, title) in zip(axes, plot_groups):

            group_mask = labels == group_id
            color = cluster_colors[group_id]

            for trajectory_id in np.flatnonzero(group_mask):

                trajectory = X_curvature[trajectory_id]

                ax.plot(trajectory[0], trajectory[1], color=color, alpha=0.12, linewidth=0.7)

            ax.scatter(-mu, 0, color="blue", s=30, label="Earth", zorder=3,)
            ax.scatter(1 - mu, 0, color="darkred", s=30, label="Moon", zorder=3,)

            ax.set_title(f"{title} || {np.count_nonzero(group_mask)} trajectories")

            ax.set_xlabel("X [LU]")
            ax.set_ylabel("Y [LU]")
            ax.set_aspect("equal")

            ax.set_box_aspect(1)


        for ax in axes[n_panels:]:
            ax.axis("off")


        figure_number = (start // max_panels + 1)

        fig.suptitle(f"Cluster groups — Figure {figure_number}")
        save_figure(fig, f"clusters_{figure_number:03d}.png")



# Plot representative trajectories
if PLOT_MEDOIDS or SAVE:

    fig_representatives, ax_representatives = plt.subplots(figsize=(10, 8), constrained_layout=True)

    for cluster_id, representative_id in enumerate(representative_ids):

        trajectory = X_curvature[representative_id]

        ax_representatives.plot(trajectory[0], trajectory[1], color="navy", alpha=0.045, linewidth=0.35, rasterized=True)

    ax_representatives.scatter(-mu, 0, color="blue", s=50, label="Earth", zorder=3)
    ax_representatives.scatter(1 - mu, 0, color="darkred", s=50, label="Moon", zorder=3)

    ax_representatives.set_xlabel("X [LU]")
    ax_representatives.set_ylabel("Y [LU]")
    ax_representatives.set_title(f"Final representative trajectories — {len(representative_ids)} medoids")
    ax_representatives.set_aspect("equal")
    ax_representatives.set_box_aspect(1)
    ax_representatives.grid(True, alpha=0.25)
    ax_representatives.legend()
    save_figure(fig_representatives, "representatives.png")



# Plot clustering in physical feature space
if SAVE or PLOT_CLUSTERS:

    features_plot = event_features[:, :3].copy()
    features_plot[:, 2] %= 360

    names = [r"$\varepsilon$ [km$^2$/s$^2$]", "e [-]", r"$\omega$ [deg]"]
    pairs = [(0, 1), (0, 2), (1, 2)]

    id_style = dict(
        fontsize=8,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )

    fig2, axes2 = plt.subplots(
        1, 3, figsize=(18, 5), constrained_layout=True
    )

    fig3 = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax3 = fig3.add_subplot(111, projection="3d")

    for cid in unique_clusters:

        values = features_plot[labels == cid]
        color = cluster_colors[cid]
        representative = features_plot[representative_ids[cid]]

        for ax, (i, j) in zip(axes2, pairs):
            ax.scatter(
                values[:, i], values[:, j],
                s=3, alpha=0.3, color=color, rasterized=True,
            )
            ax.annotate(
                str(cid),
                representative[[i, j]],
                xytext=(4, 4),
                textcoords="offset points",
                **id_style,
            )

        ax3.scatter(
            values[:, 0], values[:, 1], values[:, 2],
            s=3, alpha=0.3, color=color, depthshade=False,
        )
        ax3.text(*representative, str(cid), **id_style)

    for ax, (i, j) in zip(axes2, pairs):
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.grid(alpha=0.3)

    ax3.set_xlabel(names[0])
    ax3.set_ylabel(names[1])
    ax3.set_zlabel(names[2])

    fig2.suptitle(f"Feature projections — K={n_clusters}")
    ax3.set_title(f"Feature space — K={n_clusters}")

    save_figure(fig2, f"features_2D_K{n_clusters}.png")
    save_figure(fig3, f"features_3D_K{n_clusters}.png")

    # One additional feature-dispersion figure per cluster
    for cid in unique_clusters:

        values = features_plot[labels == cid].copy()
        color = cluster_colors[cid]
        local_names = names.copy()

        # Recenter omega to avoid artificial spreading across 0/360 degrees.
        angular_mean = np.mean(np.exp(1j * np.deg2rad(values[:, 2])))
        angle_defined = abs(angular_mean) > 1e-8

        if angle_defined:
            origin = np.rad2deg(np.angle(angular_mean))
            values[:, 2] = (values[:, 2] - origin + 180) % 360 - 180
            local_names[2] = r"$\Delta\omega$ [deg]"

        stds = np.std(values, axis=0)

        if not angle_defined:
            stds[2] = np.nan

        fig, axes = plt.subplots(
            2, 3, figsize=(15, 8), constrained_layout=True
        )

        for ax, (i, j) in zip(axes[0], pairs):
            ax.scatter(
                values[:, i], values[:, j],
                s=5, alpha=0.35, color=color, rasterized=True,
            )
            ax.set_xlabel(local_names[i])
            ax.set_ylabel(local_names[j])
            ax.grid(alpha=0.3)

        for j, ax in enumerate(axes[1]):
            ax.hist(values[:, j], bins=30, color=color, alpha=0.8)
            ax.set_xlabel(local_names[j])
            ax.set_ylabel("Number of trajectories")
            ax.set_title(f"STD = {stds[j]:.5g}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Cluster {cid} — {len(values)} trajectories")
        save_figure(fig, f"cluster_{cid:03d}_features.png")



# Show or close figures
if PLOT_CLUSTERS:
    plt.show()
else:
    plt.close("all")


if PLOT_CLUSTERS or PLOT_MEDOIDS:
    plt.show()