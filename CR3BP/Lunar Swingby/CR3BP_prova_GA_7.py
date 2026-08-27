from Celestial_Mechanics import Integration
from pathlib import Path
from multiprocessing import get_context
import time

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN


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

source_ids = valid_ids[::100]
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



trajectory_scaler = StandardScaler()
physical_scaler = StandardScaler()
pca_scaler = StandardScaler()
pca = PCA(n_components=0.99)


trajectory_features = X.reshape(len(X), -1)
trajectory_features_scaled = trajectory_scaler.fit_transform(trajectory_features)
trajectory_features_pca = pca.fit_transform(trajectory_features_scaled)
trajectory_features_pca_scaled = pca_scaler.fit_transform(trajectory_features_pca)

physical_features = selection[:, 6:]
physical_features_scaled = physical_scaler.fit_transform(physical_features)


trajectory_block = (trajectory_features_pca_scaled / np.sqrt(trajectory_features_pca_scaled.shape[1]))
physical_block = (physical_features_scaled / np.sqrt(physical_features_scaled.shape[1]))

#clustering_features = np.hstack((trajectory_block, physical_block,))
clustering_features = trajectory_features_pca

print("Original features:", trajectory_features_scaled.shape[1])
print("PCA components:", trajectory_features_pca.shape[1])
print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")



min_cluster_size = 10
min_samples = 5


clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method="eom")

labels = clusterer.fit_predict(clustering_features)
probabilities = clusterer.probabilities_



cluster_labels = labels[labels >= 0]

unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True,)

n_clusters = len(unique_clusters)
n_noise = np.count_nonzero(labels == -1)
noise_fraction = n_noise / len(labels)


print(f"\nNumber of clusters: {n_clusters}")
print(f"Noise trajectories: {n_noise}/{len(labels)} ({100 * noise_fraction:.1f} %)\n")


for cluster_id, cluster_size in zip(unique_clusters, cluster_sizes,):

    mean_probability = np.mean(probabilities[labels == cluster_id])

    print(f"Cluster {cluster_id}: {cluster_size} trajectories, mean probability = {mean_probability:.3f}")





# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")



if PLOT:
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    noise_mask = labels == -1

    ax.scatter(
        trajectory_features_pca[noise_mask, 0],
        trajectory_features_pca[noise_mask, 1],
        color="lightgray",
        s=10,
        alpha=0.5,
        label="Noise",
    )

    cmap = plt.get_cmap("turbo", max(n_clusters, 1),)

    cluster_colors = {cluster_id: cmap(color_id) for color_id, cluster_id in enumerate(unique_clusters)}


    for cluster_id in unique_clusters:

        cluster_mask = labels == cluster_id

        ax.scatter(
            trajectory_features_pca[cluster_mask, 0],
            trajectory_features_pca[cluster_mask, 1],
            color=cluster_colors[cluster_id],
            s=15,
            alpha=0.7,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("HDBSCAN clusters — PCA projection")
    ax.legend()
    plt.show()



    variance_2D = np.sum(pca.explained_variance_ratio_[:2])
    print(f"Variance represented in PCA plot: "f"{100 * variance_2D:.1f} %")



    groups = [(-1, "Noise")]

    for cluster_id in unique_clusters:
        groups.append((cluster_id, f"Cluster {cluster_id}"))



    n_panels = len(groups)

    figure_width = 18
    figure_height = 9
    figure_ratio = figure_width / figure_height

    ncols = max(1, round(np.sqrt(n_panels * figure_ratio)),)

    nrows = int(np.ceil(n_panels / ncols))


    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
        squeeze=False,
    )


    axes = axes.ravel()

    for ax, (group_id, title) in zip(axes, groups):

        group_mask = labels == group_id

        if group_id == -1:
            color = "gray"
        else:
            color = cluster_colors[group_id]

        for trajectory in X[group_mask]:

            ax.plot(
                trajectory[0],
                trajectory[1],
                color=color,
                alpha=0.12,
                linewidth=0.7,
            )

        ax.scatter(-mu, 0, color="blue", s=30, label="Earth", zorder=3,)
        ax.scatter(1 - mu, 0, color="darkred", s=30, label="Moon", zorder=3,)

        ax.set_title(f"{title} || {np.count_nonzero(group_mask)} trajectories")
        ax.set_xlabel("X [LU]")
        ax.set_ylabel("Y [LU]")
        ax.set_aspect("equal")
        ax.set_box_aspect(1)

    for ax in axes[n_panels:]:
        ax.axis("off")

    plt.show()