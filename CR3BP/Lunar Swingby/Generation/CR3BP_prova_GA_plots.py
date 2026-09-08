from multiprocessing import get_context
import time

from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np

from Celestial_Mechanics import Integration, Transformations


# Input
DATA_FILES = {
    "escape": "Escape_trajectories.txt",
    "capture": "Capture_trajectories.txt",
}

# Earth-Moon constants
G = 6.67430e-20
d = 384400.0
m1 = 5.974e24
m2 = 7.348e22
ms = 1.989e30
d_E = 1.496e8
R_E = 6378.0
R_L = 1737.0
M = m1 + m2
mu = m2 / M
n = np.sqrt(G * M / d**3)
TU = 1 / n
M_SOI = (m2 / m1) ** (2 / 5) * d
E_SOI = (m1 / ms) ** (2 / 5) * d_E

# Integration settings in normalized time units
T_FORWARD = 8 * np.pi
T_RETROGRADE = -4 * np.pi
DT = 0.0001


flyby_colors = [
    "#6A00A8",  # 1: viola
    "#00B4D8",  # 2: ciano
    "#FF8C00",  # 3: arancione
    "#2DC653",  # 4: verde
    "#FF006E",  # 5: magenta
    "#FFD60A",  # 6: giallo
]


Integrator = Integration()


def min_distance(t, x, mu):
    r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0.0, 0.0]))
    r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0.0, 0.0]))
    return min(r_earth - R_E / d, r_moon - R_L / d)


def moon_SOI_exit(t, x, mu):
    return M_SOI / d - np.linalg.norm(x[:3] - np.array([1 - mu, 0.0, 0.0]))


def earth_SOI(t, x, mu):
    return E_SOI / d - np.linalg.norm(x[:3] - np.array([-mu, 0.0, 0.0]))


def moon_SOI_entry(t, x, mu):
    return M_SOI / d - np.linalg.norm(x[:3] - np.array([1 - mu, 0.0, 0.0]))


def earth_perigee_escape(t, x, mu):
    r = x[:3] - np.array([-mu, 0.0, 0.0])
    return np.dot(r, x[3:])


def earth_perigee_capture(t, x, mu):
    r = x[:3] - np.array([-mu, 0.0, 0.0])

    if np.linalg.norm(r) > (R_E + 1000) / d:  
        return 1.0 

    return np.dot(r, x[3:])



min_distance.terminal = True
min_distance.direction = -1
moon_SOI_exit.terminal = False
moon_SOI_exit.direction = -1
moon_SOI_entry.terminal = False
moon_SOI_entry.direction = 1
earth_SOI.terminal = True
earth_SOI.direction = -1
earth_perigee_escape.terminal = True
earth_perigee_escape.direction = -1
earth_perigee_capture.terminal = True
earth_perigee_capture.direction = 1


def integrate(x0, t0, tf, events):
    return Integrator.Integrator(
        [m1, m2],
        G,
        x0,
        t0,
        tf,
        DT if tf > t0 else -DT,
        model="CR3BP",
        integrator="scipy",
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
        events=events,
    )


def to_inertial(sol):
    state = Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M)
    return Transformations.CR3BP_to_inertial(state, n, sol[0] * TU)


def integrate_trajectory(row, mode):
    t0 = row[6] / TU
    state = Transformations.inertial_to_CR3BP(row[:6], n, row[6])[:, 0]
    x0 = Transformations.SV_to_CR3BP_normalized_units(state, d, G, M)

    if mode == "escape":
        retro = integrate(
            x0,
            t0,
            T_RETROGRADE,
            (min_distance, earth_SOI, earth_perigee_escape),
        )
        forward = integrate(
            x0,
            t0,
            T_FORWARD,
            (min_distance, moon_SOI_exit, earth_SOI),
        )
        if len(forward[2][0]) or len(retro[2][0]):
            return None

        trajectory = np.concatenate(
            (to_inertial(retro)[:, ::-1], to_inertial(forward)[:, 1:]),
            axis=1,
        )
        return trajectory, len(retro[0]) - 1, len(forward[2][1])

    forward = integrate(
        x0,
        t0,
        T_FORWARD,
        (
            min_distance,
            moon_SOI_entry,
            moon_SOI_exit,
            earth_perigee_capture,
        ),
    )
    if len(forward[2][0]):
        return None

    if len(forward[2][1]):
        transition_time = forward[2][1][0]
        transition_index = np.argmin(np.abs(forward[0] - transition_time))
    else:
        transition_index = 0

    return to_inertial(forward), transition_index, len(forward[2][2])


def discrete_flyby_scale(flybys):
    levels = np.unique(flybys)
    cmap = ListedColormap([flyby_colors[(int(x)-1) % len(flyby_colors)] for x in levels])
    return cmap, BoundaryNorm(np.r_[levels-0.5, levels[-1]+0.5], cmap.N), levels


def plot_trajectories(datasets):
    theta = np.linspace(0, 2 * np.pi, 1000)
    earth = Transformations.CR3BP_to_inertial(
        [-mu * d, 0, 0, 0, 0, 0], 1, theta
    )
    moon = Transformations.CR3BP_to_inertial(
        [(1 - mu) * d, 0, 0, 0, 0, 0], 1, theta
    )

    all_Cj = np.concatenate([dataset["Cj"] for dataset in datasets])
    all_delta_v = np.concatenate([dataset["delta_v"] for dataset in datasets])
    all_flybys = np.concatenate([dataset["flybys"] for dataset in datasets])

    cmap_Cj = plt.cm.viridis
    cmap_DV = plt.cm.Spectral_r
    norm_Cj = Normalize(all_Cj.min(), all_Cj.max())
    norm_DV = Normalize(all_delta_v.min(), all_delta_v.max())
    cmap_flyby, norm_flyby, flyby_levels = discrete_flyby_scale(all_flybys)

    metrics = (
        ("Cj", cmap_Cj, norm_Cj, r"$C_J$"),
        ("delta_v", cmap_DV, norm_DV, r"$\Delta V$ [km/s]"),
        ("flybys", cmap_flyby, norm_flyby, "Number of flybys"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    soi_angle = np.linspace(0, 2 * np.pi, 500)
    earth_x, earth_y = earth[0, 0], earth[1, 0]

    for row, dataset in enumerate(datasets):
        for column, (key, cmap, norm, label) in enumerate(metrics):
            ax = axes[row, column]
            ax.plot(earth[0], earth[1], "--", color="blue", linewidth=1.5)
            ax.plot(moon[0], moon[1], "--", color="black", linewidth=1.5)
            ax.plot(
                earth_x + E_SOI * np.cos(soi_angle),
                earth_y + E_SOI * np.sin(soi_angle),
                "--",
                color="black",
                linewidth=1.5,
            )

            for trajectory, index, value in zip(
                dataset["trajectories"],
                dataset["transition_indices"],
                dataset[key],
            ):
                color = cmap(norm(value))
                ax.plot(trajectory[0], trajectory[1], color=color, linewidth=1)
                ax.scatter(
                    trajectory[0, index],
                    trajectory[1, index],
                    color=color,
                    s=6,
                    zorder=3,
                )

            ax.scatter(
                earth_x,
                earth_y,
                color="blue",
                s=45,
                label="Earth",
                zorder=4,
            )
            ax.scatter(
                moon[0, 0],
                moon[1, 0],
                color="darkred",
                s=45,
                label="Moon",
                zorder=4,
            )
            ax.set_title(
                f"{dataset['mode'].capitalize()} - Baricentric Inertial Frame"
            )
            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.axis("equal")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

    for column, (_, cmap, norm, label) in enumerate(metrics):
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes[:, column],
            label=label,
        )
        if label == "Number of flybys":
            colorbar.set_ticks(flyby_levels)

    plt.show()


def build_dataset(mode, data, results):
    accepted = [
        (result, row)
        for result, row in zip(results, data)
        if result is not None
    ]
    if not accepted:
        raise RuntimeError(f"No {mode} trajectories available for plotting.")

    trajectories = [result[0] for result, _ in accepted]
    transition_indices = [result[1] for result, _ in accepted]
    flybys = np.array([result[2] for result, _ in accepted])
    accepted_data = np.array([row for _, row in accepted])
    return {
        "mode": mode,
        "trajectories": trajectories,
        "transition_indices": transition_indices,
        "flybys": flybys,
        "Cj": accepted_data[:, 7],
        "delta_v": accepted_data[:, 9],
        "loaded": len(data),
    }


def main():
    start = time.perf_counter()
    input_data = {
        mode: np.atleast_2d(np.genfromtxt(filename, skip_header=1))
        for mode, filename in DATA_FILES.items()
    }

    with get_context("fork").Pool() as pool:
        datasets = [
            build_dataset(
                mode,
                data,
                pool.starmap(
                    integrate_trajectory,
                    ((row, mode) for row in data),
                ),
            )
            for mode, data in input_data.items()
        ]

    for dataset in datasets:
        print(
            f"{dataset['mode'].capitalize()}: "
            f"{len(dataset['trajectories'])}/{dataset['loaded']} trajectories"
        )
    print(f"Integration time: {time.perf_counter() - start:.2f} s")

    plot_trajectories(datasets)


if __name__ == "__main__":
    main()
