from Celestial_Mechanics import Integration, Transformations
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time


PLOT = True
GIF = True


# Stop at outgoing Earth SOI crossing
def earth_SOI_exit(t, state, *args):

    earth = state[0]
    spacecraft = state[2]

    r = spacecraft[:3] - earth[:3]
    v = spacecraft[3:] - earth[3:]

    return np.linalg.norm(r) >= E_SOI and np.dot(r, v) > 0


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

masses = [m1, m2, 0, ms]

#DATA_FILE = ("/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Escape_transfers.txt")
DATA_FILE = "/home/lucacecca/Astrodynamics/BCR4BP/Transfer/Escape_transfers_database_solar.txt"
database = np.atleast_2d(np.loadtxt(DATA_FILE))


SORT_BY = "E_SOI"
DESCENDING = True

columns = { "tof": 14,
            "DV1": 15,
            "DV2": 16,
            "DVtotal": 17,
            "E_SOI": 18,
            "h_moon": 19
            }


v_p_direct = np.sqrt(2 *mu_earth / (R_E + 200) + 2 * database[:, columns["E_SOI"]])
DV_direct = v_p_direct - np.sqrt(mu_earth / (R_E + 200))
performance = (DV_direct - database[:, columns["DVtotal"]])


order = np.argsort(performance, kind="stable")
selection = database[order[::-1] if DESCENDING else order]



trajectory_id = 0
dt = 0.0001 * TU


# Initialization
start_time = time.perf_counter()
Integrator = Integration()


# Load selected transfer
row = selection[trajectory_id]

t_departure = row[0]
x_departure = row[1:7]

t_manovra = row[7]
dx_manovra = row[8:14]

tof = row[14]
DV1 = row[15]
DV2 = row[16]
DVtotal = row[17]
ESOI = row[18]
h_moon = row[19]
DV_direct = (np.sqrt(2 * mu_earth / (R_E + 200) + 2 * ESOI) - np.sqrt(mu_earth / (R_E + 200)))
gain = DV_direct - DVtotal


# Earth and Moon at departure epoch
earth_departure = Transformations.CR3BP_to_inertial([-mu*d, 0, 0, 0, 0, 0], n, t_departure).ravel()
moon_departure = Transformations.CR3BP_to_inertial([(1-mu)*d, 0, 0, 0, 0, 0], n, t_departure).ravel()

# Sun at departure epoch
n_s = np.sqrt(G * (ms + M) / d_E**3)
theta_s = n_s * t_departure
sun_departure = np.array([d_E * np.cos(theta_s), d_E * np.sin(theta_s), 0, -n_s * d_E * np.sin(theta_s), n_s * d_E * np.cos(theta_s), 0])


# N-body initial state: Earth, Moon, spacecraft
state_departure = np.concatenate((earth_departure, moon_departure, x_departure, sun_departure))


# First leg: Earth departure to maneuver point
T_transfer, X_transfer = Integrator.Integrator(
    masses,
    G,
    state_departure,
    0,
    tof,
    dt,
    model="NBP",
    integrator="rebound",
    method="ias15",
    rtol=1e-9
)


# Apply maneuver to spacecraft
state_manovra = X_transfer[:, -1, :].copy()
state_manovra[2] += dx_manovra


# Time available after the maneuver
escape_duration = -t_manovra


# Second leg: maneuver point to Earth SOI
T_ETD, X_ETD = Integrator.Integrator(
    masses,
    G,
    state_manovra,
    0,
    escape_duration,
    dt,
    model="NBP",
    integrator="rebound",
    method="ias15",
    rtol=1e-9,
    stop_condition=earth_SOI_exit
)



state_ETD = X_ETD[:, -1, :].copy()


# Second leg: maneuver point to Earth SOI
T_escape, X_escape = Integrator.Integrator(
    masses,
    G,
    state_ETD,
    0,
    2 * np.pi * TU,
    dt,
    model="NBP",
    integrator="rebound",
    method="ias15",
    rtol=1e-9,
    stop_condition=earth_SOI_exit
)


# Join both legs, avoiding duplicate maneuver point
X_complete = np.concatenate((X_transfer, X_ETD, X_escape), axis=1)

maneuver_index = X_transfer.shape[1] - 1
ETD_index = X_transfer.shape[1] + X_ETD.shape[1] - 1

earth = X_complete[0]
moon = X_complete[1]
spacecraft = X_complete[2]


# Results
final_earth_distance = np.linalg.norm(spacecraft[-1, :3] - earth[-1, :3])

print("\n=== COMPLETE TRANSFER ===")
print(f"Departure time: {t_departure / 86400:.2f} days")
print(f"Maneuver time:  {t_manovra / 86400:.2f} days")
print(f"Time of flight: {tof / 86400:.2f} days")
print(f"Moon altitude:  {h_moon:.1f} km")
print(f"DV1:            {DV1:.3f} km/s")
print(f"DV2:            {DV2:.3f} km/s")
print(f"DV total:       {DVtotal:.3f} km/s")
print(f"ESOI:           {ESOI:.3f} km²/s²")
print(f"DV_direct:      {DV_direct:.3f} km/s")
print(f"DV_gain:        {gain:.3f} km/s ({gain/DV_direct * 100:.1f} %)")



elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s")




if PLOT:
    plt.figure(figsize=(10, 10))

    plt.plot(
        moon[:, 0] - earth[:, 0],
        moon[:, 1] - earth[:, 1],
        color="darkred",
        linestyle="--",
        label="Moon"
    )

    plt.plot(
        spacecraft[:maneuver_index + 1, 0] - earth[:maneuver_index + 1, 0],
        spacecraft[:maneuver_index + 1, 1] - earth[:maneuver_index + 1, 1],
        color="green",
        linewidth=2,
        label="Transfer"
    )

    plt.plot(
        spacecraft[maneuver_index:ETD_index + 1, 0] - earth[maneuver_index:ETD_index + 1, 0],
        spacecraft[maneuver_index:ETD_index + 1, 1] - earth[maneuver_index:ETD_index + 1, 1],
        color="blue",
        linewidth=2,
        label="ETD"
    )

    plt.plot(
        spacecraft[ETD_index:, 0] - earth[ETD_index:, 0],
        spacecraft[ETD_index:, 1] - earth[ETD_index:, 1],
        color="red",
        linewidth=2,
        label="Escape"
    )

    plt.scatter(
        spacecraft[0, 0] - earth[0, 0],
        spacecraft[0, 1] - earth[0, 1],
        color="green",
        s=50,
        zorder=50,
        label="Departure"
    )

    plt.scatter(
        spacecraft[maneuver_index, 0] - earth[maneuver_index, 0],
        spacecraft[maneuver_index, 1] - earth[maneuver_index, 1],
        color="blue",
        s=50,
        zorder=50,
        label="Maneuver"
    )


    plt.scatter(
        spacecraft[ETD_index, 0] - earth[ETD_index, 0],
        spacecraft[ETD_index, 1] - earth[ETD_index, 1],
        color="red",
        s=50,
        zorder=50,
        label="ETD"
    )


    plt.scatter(
        spacecraft[-1, 0] - earth[-1, 0],
        spacecraft[-1, 1] - earth[-1, 1],
        color="black",
        s=50,
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
    plt.legend()
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
        "/home/lucacecca/Astrodynamics/BCR4BP/Transfer/Escape.gif",
        writer=animation.PillowWriter(fps=30),
        dpi=100
    )

    plt.close(fig)

    print("GIF saved.")