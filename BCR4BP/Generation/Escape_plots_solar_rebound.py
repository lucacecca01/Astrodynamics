from Celestial_Mechanics import Integration, Transformations
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time


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
selection = database[order[::-1] if DESCENDING else order]


valid_mask = ((selection[:, columns["h_min_moon"]] >= 0)
                & (selection[:, columns["h_min_earth"]] >= 0)
                & (selection[:, columns["E_min"]] <= 0)
                & (selection[:, columns["E_SOI"]] > 0)
                & (selection[:, columns["n_flybys"]] > 0)
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
    plt.figure(figsize=(10, 10))

for trajectory_id in trajectory_ids:
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

        if phase == 1:
            E_out = earth_energy(X_ETD[:, -1])

        elif phase == 2:
            peri_end = len(T_ETD)

        elif phase == 3:
            E_in = earth_energy(X_ETD[:, -1])

            if E_out > E_in:
                accepted = True
                break

        phase = 1 if phase == 3 else phase + 1

   
    end = peri_end if accepted else 1
    T_ETD = T_ETD[:end]
    X_ETD = X_ETD[:, :end]

    T_escape, X_escape, te, xe = Integrator.Integrator(
    masses, G, x0, 0, 4 * np.pi, dt,
    model="BCR4BP",
    integrator="scipy",
    method="DOP853",
    rtol=1e-9,
    atol=1e-12,
    events=earth_SOI_exit,
    r_s=r_s,
    theta_s_0=0
)

    escaped = len(te[0]) > 0

    if escaped:
        keep = T_escape < te[0][0]
        T_escape = np.r_[T_escape[keep], te[0][0]]
        X_escape = np.column_stack((X_escape[:, keep], xe[0][0]))

    T_complete = np.r_[T_ETD[::-1], T_escape[1:]]
    X_complete = np.column_stack((X_ETD[:, ::-1], X_escape[:, 1:]))
    ETD_index = len(T_ETD) - 1

    X_geo = X_complete.copy()
    X_geo[0] += mu

    time_seconds = T_complete * TU

    spacecraft = Transformations.CR3BP_to_inertial(Transformations.CR3BP_normalized_units_to_SV(X_geo, d, G, M), n, time_seconds).T
    moon = Transformations.CR3BP_to_inertial([d, 0, 0, 0, 0, 0], n, time_seconds).T
    earth = np.zeros_like(spacecraft)


    final_earth_distance = np.linalg.norm(spacecraft[-1, :3] - earth[-1, :3])
    distance_moon = np.linalg.norm(spacecraft[:, :3] - moon[:, :3], axis=1,)
    inside = distance_moon < M_SOI


    if PRINT:
        n_flybys = int(np.count_nonzero(~inside[:-1] & inside[1:]))
        E_final = 0.5 * np.linalg.norm(spacecraft[-1, 3:] - earth[-1, 3:])**2 - mu_earth / final_earth_distance
        print(f"Ingressi nella SOI lunare: {n_flybys}")
        print(f"Energia finale: {E_final:.3f} km²/s²")

        if np.abs(E_final - row[7]) > 1e-1:
            print(f"Warning: final energy differs from database value by {np.abs(E_final - row[7]):.3f} km²/s²")





    if PLOT:

        plt.plot(
            moon[:, 0] - earth[:, 0],
            moon[:, 1] - earth[:, 1],
            color="darkred",
            linestyle="--",
            label="Moon"
        )

        

        plt.plot(
            spacecraft[:, 0] - earth[:, 0],
            spacecraft[:, 1] - earth[:, 1],
            color=cmap(norm(row[columns[COLOR_BY]])),
            alpha=0.5,
            linewidth=1,
        )

        plt.scatter(
            spacecraft[0, 0] - earth[0, 0],
            spacecraft[0, 1] - earth[0, 1],
            color="green",
            s=10,
            zorder=50,
            label="Departure"
        )



        plt.scatter(
            spacecraft[ETD_index, 0] - earth[ETD_index, 0],
            spacecraft[ETD_index, 1] - earth[ETD_index, 1],
            color="red",
            s=10,
            zorder=50,
            label="ETD"
        )


        plt.scatter(
            spacecraft[-1, 0] - earth[-1, 0],
            spacecraft[-1, 1] - earth[-1, 1],
            color="black",
            s=10,
            zorder=50,
            label="Earth SOI"
        )

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
        plt.title("Complete trajectory — Earth inertial frame")
        plt.axis("equal")

        if trajectory_id == trajectory_ids[0]:
            plt.legend()

        plt.tight_layout()

print('Max E_SOI:', np.max(x0_selection[trajectory_ids, columns["E_SOI"]]))
print('Min E_SOI:', np.min(x0_selection[trajectory_ids, columns["E_SOI"]]))

elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s")


if PLOT:
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=plt.gca(),
        label=COLOR_BY
    )
    plt.tight_layout()
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