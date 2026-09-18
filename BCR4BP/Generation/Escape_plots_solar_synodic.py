from Celestial_Mechanics import Integration, Transformations
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
import time


flyby_colors = [
    "#6A00A8",  # 1: viola
    "#00B4D8",  # 2: ciano
    "#FF8C00",  # 3: arancione
    "#2DC653",  # 4: verde
    "#FF006E",  # 5: magenta
    "#FFD60A",  # 6: giallo
    "#FF1FFF",  # 7: fucsia
    "#000DFF",  # 8: blu
    "#5F2106",  # 9: marrone
    "#000000",  # 10: nero
]


PLOT = True
GIF = False
PRINT = False


def earth_SOI_exit(t, x, mu, *args):
    return np.linalg.norm(x[:3] - [-mu, 0, 0]) - E_SOI / d


def moon_encounter(t, x, mu, *args):
    return np.linalg.norm(x[:3] - [1 - mu, 0, 0]) - M_SOI / d


def lunar_perilune(t, x, mu, *args):
    return np.dot(x[:3] - [1 - mu, 0, 0], x[3:])


def earth_energy(x):
    r = x[:3] + np.array([mu, 0, 0])
    v = x[3:] + np.cross([0, 0, 1], r)
    return (0.5 * np.dot(v, v) - (1 - mu) / np.linalg.norm(r)) * G * M / d
   

# Constants
G = 6.67430e-20
d = 384400
m1 = 5.974e24
m2 = 7.348e22
ms = 1.989e30
d_E = 1.496e8
R_E = 6378                   # Earth radius in km
R_L = 1737                   # Moon radius in km
mu_earth = G * m1

M = m1 + m2
mu = m2 / M
n = np.sqrt(G * M / d**3)
TU = 1 / n

E_SOI = (m1 / ms)**(2/5) * d_E
M_SOI = (m2 / m1)**(2/5) * d

masses = [m1, m2, ms]
r_s = d_E / d

#DATA_FILE = ("/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Escape_transfers.txt")
DATA_FILE = "/home/lucacecca/Astrodynamics/BCR4BP/Generation/Escape_initial_conditions_CR3BP_solar_2.txt"
database = np.atleast_2d(np.loadtxt(DATA_FILE))


SORT_BY = "E_SOI"
DESCENDING = True

columns = {
    "Cj": 6,
    "E_SOI": 7,
    "h_min_moon": 8,
    "h_min_earth": 9,
    "n_flybys": 10,
    "E_min": 11
}

order = np.argsort(database[:, columns[SORT_BY]], kind="stable")
#order = np.argsort(database[:, columns["E_SOI"]] - database[:, columns["E_min"]], kind="stable")

selection = database[order[::-1] if DESCENDING else order]
selection = database[order[::-1] if DESCENDING else order]


valid_mask = ((selection[:, columns["h_min_moon"]] >= 0)
                & (selection[:, columns["h_min_earth"]] >= 0)
                & (selection[:, columns["E_min"]] <= 0)
                & (selection[:, columns["E_SOI"]] > 0)
                & (selection[:, columns["n_flybys"]] == 1)
)

valid_ids = np.where(valid_mask)[0]

source_ids = valid_ids[::1]
x0_selection = selection[source_ids, :]

print(f"Total:                     {len(database)}")
print(f"Accepted initial filters:  {len(x0_selection)}\n")




trajectory_ids = np.arange(0, 1000, 1) 
dt = 0.001

COLOR_BY = "E_SOI"

values = x0_selection[trajectory_ids, columns[COLOR_BY]]
vmin, vmax = values.min(), values.max()

if vmin == vmax:
    vmin, vmax = vmin - 0.5, vmax + 0.5

norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("viridis")


# Initialization
start_time = time.perf_counter()
Integrator = Integration()



earth_SOI_exit.terminal = True
earth_SOI_exit.direction = 1

moon_encounter.terminal = True
moon_encounter.direction = 1

lunar_perilune.terminal = True
lunar_perilune.direction = -1


if PLOT:
    fig_traj, axes_traj = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    axes_traj = axes_traj.ravel()

    titles = ("1 flyby", "2 flyby", "3 flyby", "4 flyby", "5 flyby", "> 5 flyby")

    for ax, title in zip(axes_traj, titles):
        ax.set_title(title)
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")

tof_days = np.full(len(trajectory_ids), np.nan)

for i, trajectory_id in enumerate(trajectory_ids):
    row = x0_selection[trajectory_id]

    if PRINT:
        print("\n=== COMPLETE TRAJECTORY ===")
        print(f"Cj:             {row[6]:.2f}")
        print(f"E_SOI:          {row[7]:.3f} km²/s²")
        print(f"h_min_moon:     {row[8]:.0f} km")
        print(f"h_min_earth:    {row[9]:.0f} km")
        print(f"N_flybys:       {int(row[10])}")
        print(f"E_earth_min:    {row[11]:.3f} km²/s²\n")


    x0 = row[:6].copy()

    T_ETD = np.array([0.0])
    X_ETD = x0[:, None]

    accepted = False
    phase = 0

    phases = (
        (moon_encounter, 1),   # Esclude l'incontro dell'ETD.
        (moon_encounter, -1),  # Uscita del flyby nel moto in avanti.
        (lunar_perilune, -1),  # Perilunio.
        (moon_encounter, 1)    # Ingresso del flyby nel moto in avanti.
    )

    while T_ETD[-1] > -4 * np.pi:
        event, direction = phases[phase]
        event.direction = direction
        event.terminal = True

        t, x, te, xe = Integrator.Integrator(
            masses, G, X_ETD[:, -1], T_ETD[-1], -4 * np.pi, -dt,
            model="BCR4BP",
            integrator="scipy",
            method="DOP853",
            rtol=1e-9,
            atol=1e-12,
            events=(event, earth_SOI_exit),
            r_s=r_s, 
            theta_s_0=0
        )

        if len(te[0]) == 0:
            break

        keep = (t < T_ETD[-1]) & (t > te[0][0])
        T_ETD = np.r_[T_ETD, t[keep], te[0][0]]
        X_ETD = np.column_stack((X_ETD, x[:, keep], xe[0][0]))

        if phase == 2:
            peri_end = len(T_ETD)
            accepted = True
            break

        phase += 1

   
    end = peri_end if accepted else 1
    T_ETD = T_ETD[:end]
    X_ETD = X_ETD[:, :end]

    moon_encounter.terminal = False
    moon_encounter.direction = 1

    lunar_perilune.terminal = False
    lunar_perilune.direction = 1

    T_escape, X_escape, te, xe = Integrator.Integrator(
    masses, G, x0, 0, 4 * np.pi, dt,
    model="BCR4BP",
    integrator="scipy",
    method="DOP853",
    rtol=1e-9,
    atol=1e-12,
    events=(earth_SOI_exit, moon_encounter, lunar_perilune),
    r_s=r_s,
    theta_s_0=0
)

    escaped = len(te[0]) > 0

    if escaped:
        keep = T_escape < te[0][0]
        T_escape = np.r_[T_escape[keep], te[0][0]]
        X_escape = np.column_stack((X_escape[:, keep], xe[0][0]))

    n_forward = len(te[1])  # Include l'incontro ETD.

    if not escaped or n_forward not in (1, 2):
        continue
    if n_forward == 1 and not accepted and row[columns["n_flybys"]] > 1:
        continue

    if n_forward == 2:
        T_ETD = T_ETD[:1]
        X_ETD = X_ETD[:, :1]

    T_complete = np.r_[T_ETD[::-1], T_escape[1:]]
    X_complete = np.column_stack((X_ETD[:, ::-1], X_escape[:, 1:]))
    ETD_index = len(T_ETD) - 1

    if escaped:
        tof_days[i] = (T_complete[-1] - T_complete[0]) * TU / 86400

    X_geo = X_complete.copy()
    X_geo[0] += mu

    time_seconds = T_complete * TU

    spacecraft = Transformations.CR3BP_normalized_units_to_SV(X_geo, d, G, M).T
    #moon = Transformations.CR3BP_to_inertial([d, 0, 0, 0, 0, 0], n, time_seconds).T
    #earth = np.zeros_like(spacecraft)


    # final_earth_distance = np.linalg.norm(spacecraft[-1, :3] - earth[-1, :3])
    # distance_moon = np.linalg.norm(spacecraft[:, :3] - moon[:, :3], axis=1,)
    # inside = distance_moon < M_SOI


    # if PRINT:
    #     n_flybys = int(np.count_nonzero(~inside[:-1] & inside[1:]))
    #     E_final = 0.5 * np.linalg.norm(spacecraft[-1, 3:] - earth[-1, 3:])**2 - mu_earth / final_earth_distance
    #     print(f"Ingressi nella SOI lunare: {n_flybys}")
    #     print(f"Energia finale: {E_final:.3f} km²/s²")

    #     if np.abs(E_final - row[7]) > 1e-1:
    #         print(f"Warning: final energy differs from database value by {np.abs(E_final - row[7]):.3f} km²/s²")





    if PLOT:

        n_fb = int(row[columns["n_flybys"]])
        group = min(n_fb - 1, 5)
        ax = axes_traj[group]
        plt.sca(ax)

        plt.scatter(
            d,
            0,
            color="darkred",
            s=25,
            label="Moon"
        )

        

        plt.plot(
            spacecraft[:, 0],
            spacecraft[:, 1],
            color=cmap(norm(row[columns[COLOR_BY]])),
            alpha=0.2,
            linewidth=0.8,
        )

        plt.scatter(
            spacecraft[0, 0],
            spacecraft[0, 1],
            color='green',
            s=10,
            zorder=50,
            label="Departure"
        )



        plt.scatter(
            spacecraft[ETD_index, 0],
            spacecraft[ETD_index, 1],
            color="red",
            s=10,
            zorder=60,
            label="ETD"
        )

        verso = -1 if lunar_perilune(0, x0, mu) > 0 else 1

        lunar_perilune.terminal = True
        lunar_perilune.direction = verso
        moon_encounter.terminal = True
        moon_encounter.direction = 1

        _, _, te_etd, xe_etd = Integrator.Integrator(
            masses, G, x0, 0, verso * 4 * np.pi, verso * dt,
            model="BCR4BP",
            integrator="scipy",
            method="DOP853",
            rtol=1e-9,
            atol=1e-12,
            events=(lunar_perilune, moon_encounter),
            r_s=r_s,
            theta_s_0=0
        )

        if len(te_etd[0]) > 0:
            peri = xe_etd[0][0]

            ax.scatter(
                (peri[0] + mu) * d, peri[1] * d,
                color="magenta",
                marker="x",
                s=35,
                zorder=70,
                label="Perilunio ETD"
            )

        # plt.scatter(
        #     spacecraft[-1, 0] - earth[-1, 0],
        #     spacecraft[-1, 1] - earth[-1, 1],
        #     color="black",
        #     s=10,
        #     zorder=50,
        #     label="Earth SOI"
        # )

        plt.scatter(0, 0, color="purple", s=25, label="Earth")

        barycentric_soi = plt.Circle(
            (0, 0),
            E_SOI,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        plt.gca().add_patch(barycentric_soi)

        plt.xlabel("X [km]")
        plt.ylabel("Y [km]")
        plt.axis("equal")

        if ax.get_legend() is None:
            plt.legend()


kept = np.isfinite(tof_days)
trajectory_ids = trajectory_ids[kept]
tof_days = tof_days[kept]

if trajectory_ids.size == 0:
    raise SystemExit("Nessuna traiettoria con esattamente due flyby.")


print('Max E_SOI:', np.max(x0_selection[trajectory_ids, columns["E_SOI"]]))
print('Min E_SOI:', np.min(x0_selection[trajectory_ids, columns["E_SOI"]]))

elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s")


if PLOT:
    data = x0_selection[trajectory_ids]
    n_fb = data[:, columns["n_flybys"]].astype(int)


    ticks_fb = np.arange(n_fb.min(), n_fb.max() + 1)
    bounds_fb = np.r_[ticks_fb - 0.5, ticks_fb[-1] + 0.5]

    cmap_fb = ListedColormap([flyby_colors[k - 1] for k in ticks_fb])
    norm_fb = BoundaryNorm(bounds_fb, cmap_fb.N)

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9),
        sharey=True, constrained_layout=True
    )
    axes = axes.ravel()

    parameters = [
        (n_fb, "Numero di flyby"),
        (data[:, columns["E_min"]], "Energia terrestre minima [km²/s²]"),
        (data[:, columns["h_min_earth"]], "Altezza terrestre minima [km]"),
        (data[:, columns["h_min_moon"]], "Altezza lunare minima [km]"),
        (data[:, columns["Cj"]], "Cj"),
        (tof_days, "TOF flyby/ETD → SOI terrestre [giorni]")
    ]

    for ax, (x_values, label) in zip(axes, parameters):
        ax.scatter(
            x_values,
            data[:, columns["E_SOI"]],
            c=n_fb,
            cmap=cmap_fb,
            norm=norm_fb,
            s=12,
            alpha=0.7
        )
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)

    axes[0].set_xticks(ticks_fb)
    axes[0].set_ylabel("E_SOI [km²/s²]")
    axes[3].set_ylabel("E_SOI [km²/s²]")

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_fb, cmap=cmap_fb),
        ax=axes.tolist(),
        ticks=ticks_fb,
        label="Numero di flyby"
    )

    # Colorbar della figura delle traiettorie.
    fig_traj.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes_traj.tolist(),
        label=COLOR_BY
    )

    plt.show()


if GIF:

    moon = moon - earth
    spacecraft = spacecraft - earth
    earth = np.zeros_like(earth)

    frame_num = min(400, len(earth))
    frame_indices = np.linspace(
        0,
        len(earth) - 1,
        frame_num,
        dtype=int
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    point1, = ax.plot(
        [], [],
        "o",
        color="blue",
        label="Earth"
    )

    point2, = ax.plot(
        [], [],
        "o",
        color="darkred",
        label="Moon"
    )

    point3, = ax.plot(
        [], [],
        "o",
        color="green",
        label="Spacecraft"
    )

    line1, = ax.plot([], [], "--", color="blue", linewidth=1)
    line2, = ax.plot([], [], "--", color="darkred", linewidth=1)
    line3, = ax.plot([], [], color="green", linewidth=1)

    all_positions = np.vstack((
        earth[:, :2],
        moon[:, :2],
        spacecraft[:, :2]
    ))

    limit = 1.05 * np.max(np.abs(all_positions))

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_title("Earth–Moon system")
    ax.legend()

    def update(frame):

        index = frame_indices[frame]

        point1.set_data(
            [earth[index, 0]],
            [earth[index, 1]]
        )

        point2.set_data(
            [moon[index, 0]],
            [moon[index, 1]]
        )

        point3.set_data(
            [spacecraft[index, 0]],
            [spacecraft[index, 1]]
        )

        line1.set_data(
            earth[:index + 1, 0],
            earth[:index + 1, 1]
        )

        line2.set_data(
            moon[:index + 1, 0],
            moon[:index + 1, 1]
        )

        line3.set_data(
            spacecraft[:index + 1, 0],
            spacecraft[:index + 1, 1]
        )

        return point1, point2, point3, line1, line2, line3

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=30,
        blit=True
    )

    ani.save(
        "/home/lucacecca/Astrodynamics/BCR4BP/Transfer/Escape_prova.gif",
        writer=animation.PillowWriter(fps=30),
        dpi=100
    )

    plt.close(fig)

    print("GIF saved.")