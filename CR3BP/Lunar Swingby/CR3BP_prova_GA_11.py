from Celestial_Mechanics import Integration
from pathlib import Path
from multiprocessing import get_context
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


SAVE = True
PLOT_CLUSTERS = True
PLOT_MEDOIDS = True



#Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)

    return E_SOI / d - r_earth



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

    solution_b = integrate_one(x0, T_min, T_max_b, dt_b, earth_SOI_exit)

    sampled_b = sample_branch(solution_b)

    del solution_b


    solution_f = integrate_one(x0, T_min, T_max_f, dt_f, earth_SOI_exit)

    sampled_f = sample_branch(solution_f)

    del solution_f


    assert np.allclose(sampled_b[:, 0], sampled_f[:, 0])

    return np.concatenate((sampled_b[:, ::-1], sampled_f[:, 1:]), axis=1)



# Define sample function based on curvature
def sample_branch(sol):

    t = sol.t
    trajectory = sol.y

    derivative = Integrator.CR3BP_ODE(0, trajectory, mu)

    velocity = derivative[:3]
    acceleration = derivative[3:]

    v_squared = np.sum(velocity**2, axis=0)

    cross_norm = np.linalg.norm(np.cross(velocity, acceleration, axis=0), axis=0)

    curvature_rate = np.divide(cross_norm, v_squared, out=np.zeros_like(v_squared), where=v_squared > 1e-14)

    progress = np.abs(t - t[0])

    cumulative_curvature = cumulative_trapezoid(curvature_rate, x=progress, initial=0)

    cumulative_curvature = np.maximum.accumulate(cumulative_curvature)

    cumulative_curvature, unique_ids = np.unique(cumulative_curvature, return_index=True)

    if cumulative_curvature[-1] <= 1e-14:

        target_times = np.linspace(t[0], t[-1], N_branch)

    else:

        targets = np.linspace(0, cumulative_curvature[-1], N_branch)
        target_times = np.interp(targets, cumulative_curvature, t[unique_ids])

        target_times[0] = t[0]
        target_times[-1] = t[-1]

    return sol.sol(target_times)



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
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Escape_initial_conditions_CR3BP.txt"

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

N_branch = 500



# Define events
earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1




# Parallel integration backward and forward
with get_context("fork").Pool() as pool:

    X_curvature = pool.map(
        integrate_and_sample_one,
        x0_selection,
        chunksize=10,
    )

X_curvature = np.stack(X_curvature)

assert np.allclose(X_curvature[:, :, N_branch - 1], x0_selection)





# Compute clustering features
sampled_position = X_curvature[:, :2]
sampled_velocity = X_curvature[:, 3:5]

sampled_speed = np.linalg.norm(sampled_velocity, axis=1, keepdims=True)
unit_tangent = np.divide(sampled_velocity, sampled_speed, out=np.zeros_like(sampled_velocity), where=sampled_speed > 1e-14)

backward_position = sampled_position[:, :, :N_branch]
forward_position = sampled_position[:, :, N_branch - 1:]

backward_tangent = unit_tangent[:, :, :N_branch]
forward_tangent = unit_tangent[:, :, N_branch - 1:]

backward_position = backward_position.transpose(0, 2, 1).reshape(len(X_curvature), -1)
forward_position = forward_position.transpose(0, 2, 1).reshape(len(X_curvature), -1)

backward_tangent = backward_tangent.transpose(0, 2, 1).reshape(len(X_curvature), -1)
forward_tangent = forward_tangent.transpose(0, 2, 1).reshape(len(X_curvature), -1)

backward_position = normalize_block(backward_position)
forward_position = normalize_block(forward_position)

backward_tangent = normalize_block(backward_tangent)
forward_tangent = normalize_block(forward_tangent)


physical_weight  = 0.4

physical_features = selection[:, [columns["E_SOI"], columns["h_min_moon"], columns["h_perigee"]]]

physical_mean = np.mean(physical_features, axis=0)
physical_std = np.std(physical_features, axis=0)

physical_features = (physical_features - physical_mean) / physical_std
physical_features /= np.sqrt(physical_features.shape[1])


n_geometric_features = sum(block.shape[1]
    for block in (backward_position, backward_tangent, forward_position, forward_tangent))

clustering_features = np.hstack((backward_position, backward_tangent, forward_position, forward_tangent, physical_weight * physical_features))

geometric_features = clustering_features[:, :n_geometric_features]

del backward_position, forward_position
del backward_tangent, forward_tangent
del sampled_position, sampled_velocity
del sampled_speed, unit_tangent









# Perform farthest point selection
max_representatives = 1000

representative_ids, labels, radius_history, minimum_distances = (
    farthest_point_selection(
        geometric_features,
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





# Refine farthest-point clusters with KMeans
kmeans = KMeans(
    n_clusters=len(farthest_representative_ids),
    init=clustering_features[farthest_representative_ids],
    n_init=1,
    max_iter=300,
    tol=1e-4,
    algorithm="lloyd",
)

kmeans_labels = kmeans.fit_predict(clustering_features)

representative_ids = np.empty(len(kmeans.cluster_centers_), dtype=int)

for cluster_id, cluster_center in enumerate(kmeans.cluster_centers_):

    cluster_ids = np.flatnonzero(kmeans_labels == cluster_id)

    distances_squared = np.sum((clustering_features[cluster_ids] - cluster_center)**2, axis=1)

    representative_ids[cluster_id] = cluster_ids[np.argmin(distances_squared)]


labels, _ = pairwise_distances_argmin_min(clustering_features, clustering_features[representative_ids])

minimum_distances = np.linalg.norm(geometric_features - geometric_features[representative_ids[labels]], axis=1)

representative_source_ids = source_ids[representative_ids]


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

cluster_variances = [np.mean(np.var(geometric_features[labels == cluster_id], axis=0))
    for cluster_id in unique_clusters]

mean_variance = np.average(cluster_variances, weights=cluster_sizes)


for cluster_id in unique_clusters:

    cluster_mask = labels == cluster_id
    cluster_features = geometric_features[cluster_mask]

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
if PLOT_CLUSTERS:

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



# Plot representative trajectories
if PLOT_MEDOIDS:

    representative_numbers = np.arange(1, len(radius_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(representative_numbers, radius_history, color="blue")

    ax.set_xlabel("Number of representatives")
    ax.set_ylabel("Maximum normalized distance")
    ax.set_title("Database covering radius")
    ax.grid(True)


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



# Save figures
if SAVE:
    plot_directory = (Path(__file__).resolve().parent / f"Clusters/GA_11_plots_K{len(representative_ids)}")
    
    plot_directory.mkdir(parents=True, exist_ok=True)
    
    for figure_id in plt.get_fignums():
    
        figure = plt.figure(figure_id)
    
        figure.savefig(plot_directory / f"figure_{figure_id:03d}.png", dpi=200)




if PLOT_CLUSTERS or PLOT_MEDOIDS:
    plt.show()