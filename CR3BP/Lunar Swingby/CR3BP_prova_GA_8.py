from Celestial_Mechanics import Integration
from pathlib import Path
from multiprocessing import get_context
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


SAVE = False
PLOT = True



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
    )

    return sol








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

source_ids = valid_ids[::10]
selection = database[source_ids, :]
x0_selection = selection[:, :6]


print(f"Database trajectories: {len(database)}")
print(f"Valid trajectories:    {len(valid_ids)}")
print(f"Selected trajectories: {len(x0_selection)}")


# Initialize the packages
start_time = time.perf_counter()
Integrator = Integration()



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0
T_max = -4 * np.pi
dt = -0.01


# Define events
earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1




# Parallel integration
args = [(x0, T_min, T_max, dt, earth_SOI_exit) for x0 in x0_selection]

with get_context("fork").Pool() as pool:

    results = pool.starmap(
        integrate_one,
        args,
        chunksize=10,
    )


T = np.stack([result[0] for result in results])
X = np.stack([result[1] for result in results])





# Compute curvature
state_derivative = Integrator.CR3BP_ODE(0, X.transpose(1, 0, 2), mu,).transpose(1, 0, 2)

velocity = state_derivative[:, :3]
acceleration = state_derivative[:, 3:]
speed = np.linalg.norm(velocity, axis=1)

velocity_cross_acceleration = np.cross(velocity, acceleration, axis=1,)
curvature = (np.linalg.norm(velocity_cross_acceleration, axis=1) / speed**3)


curvature_integrand = curvature * speed

cumulative_curvature = cumulative_trapezoid(curvature_integrand, x=-T, axis=1, initial=0)
curvature_total = cumulative_curvature[:, -1]




# Sample every trajectory using the same number of curvature points
N_a = 5
X_curvature = []

delta_curvature = np.pi / N_a
N_common = (int(np.ceil(np.max(curvature_total) / delta_curvature)) + 1)


for trajectory, cumulative, total in zip(X, cumulative_curvature, curvature_total,):

    targets = np.linspace(0, total, N_common)

    sampled_trajectory = np.vstack([np.interp(targets, cumulative, component) for component in trajectory])

    X_curvature.append(sampled_trajectory)



X_curvature = np.stack(X_curvature)

sampled_position = X_curvature[:, :3]
sampled_velocity = X_curvature[:, 3:]

sampled_speed = np.linalg.norm(sampled_velocity, axis=1, keepdims=True)
unit_tangent = sampled_velocity / sampled_speed

shape_features = unit_tangent.transpose(0, 2, 1).reshape(len(X_curvature), -1)
position_features = sampled_position.transpose(0, 2, 1).reshape(len(X_curvature), -1,)

shape_scaled = StandardScaler().fit_transform(shape_features)
position_scaled = StandardScaler().fit_transform(position_features)

n_features = shape_scaled.shape[1]

clustering_features = np.hstack((shape_scaled, position_scaled)) / np.sqrt(n_features)


distance_threshold = 2

if len(X) == 1:

    labels = np.zeros(1, dtype=int)

else:

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward",
    )

    labels = clusterer.fit_predict(clustering_features)









unique_clusters, cluster_sizes = np.unique(labels, return_counts=True)

n_clusters = len(unique_clusters)
n_noise = np.count_nonzero(labels == -1)
noise_fraction = n_noise / len(labels)


print(f"\nNumber of clusters: {n_clusters}")
print(f"Noise trajectories: {n_noise}/{len(labels)} ({100 * noise_fraction:.1f} %)\n")



for cluster_id, cluster_size in zip(unique_clusters, cluster_sizes,):
    print(f"Cluster {cluster_id}: {cluster_size} trajectories")





# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")



if PLOT:
    # fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # noise_mask = labels == -1

    # ax.scatter(
    #     trajectory_features_pca[noise_mask, 0],
    #     trajectory_features_pca[noise_mask, 1],
    #     color="lightgray",
    #     s=10,
    #     alpha=0.5,
    #     label="Noise",
    # )

    cmap = plt.get_cmap("turbo", max(n_clusters, 1),)

    cluster_colors = {cluster_id: cmap(color_id) for color_id, cluster_id in enumerate(unique_clusters)}


    # for cluster_id in unique_clusters:

    #     cluster_mask = labels == cluster_id

    #     ax.scatter(
    #         trajectory_features_pca[cluster_mask, 0],
    #         trajectory_features_pca[cluster_mask, 1],
    #         color=cluster_colors[cluster_id],
    #         s=15,
    #         alpha=0.7,
    #         label=f"Cluster {cluster_id}",
    #     )

    # ax.set_xlabel("PCA component 1")
    # ax.set_ylabel("PCA component 2")
    # ax.set_title("HDBSCAN clusters — PCA projection")
    # ax.legend()
    # plt.show()



    # variance_2D = np.sum(pca.explained_variance_ratio_[:2])
    # print(f"Variance represented in PCA plot: "f"{100 * variance_2D:.1f} %")



    groups = []

    for cluster_id in unique_clusters:
        groups.append((cluster_id, f"Cluster {cluster_id}"))



    max_panels = 20
    max_columns = 4


    for start in range(0, len(groups), max_panels):

        plot_groups = groups[
            start:start + max_panels
        ]

        n_panels = len(plot_groups)

        ncols = min(
            max_columns,
            n_panels,
        )

        nrows = int(
            np.ceil(n_panels / ncols)
        )


        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                3 * ncols,
                3 * nrows,
            ),
            constrained_layout=True,
            squeeze=False,
        )

        axes = axes.ravel()


        for ax, (group_id, title) in zip(
            axes,
            plot_groups,
        ):

            group_mask = labels == group_id
            color = cluster_colors[group_id]

            for trajectory in X[group_mask]:

                ax.plot(
                    trajectory[0],
                    trajectory[1],
                    color=color,
                    alpha=0.12,
                    linewidth=0.7,
                )

            ax.scatter(
                -mu,
                0,
                color="blue",
                s=30,
                label="Earth",
                zorder=3,
            )

            ax.scatter(
                1 - mu,
                0,
                color="darkred",
                s=30,
                label="Moon",
                zorder=3,
            )

            ax.set_title(
                f"{title} || "
                f"{np.count_nonzero(group_mask)} trajectories"
            )

            ax.set_xlabel("X [LU]")
            ax.set_ylabel("Y [LU]")
            ax.set_aspect("equal")
            ax.set_box_aspect(1)


        for ax in axes[n_panels:]:
            ax.axis("off")


        figure_number = (
            start // max_panels
            + 1
        )

        fig.suptitle(
            f"Cluster groups — Figure {figure_number}"
        )


    plt.show()