from Celestial_Mechanics import Integration, Transformations
import numpy as np
from multiprocessing import get_context
import time
from matplotlib import pyplot as plt


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




# Define integration function
def integrate_one(x0, T_min, T_max, dt, events=None):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=1e-9,
        events=events,
    )

    return sol



# Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)

    return E_SOI / d - r_earth



earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1



# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Escape_initial_conditions_CR3BP.txt"



# Initialize the integrator
start_time = time.perf_counter()
Integrator = Integration()



# Initialize lists to store results
T = []
X = []



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0
T_max = -4 * np.pi
dt = -0.001



r_p = (R_E + 200) / d
r_a_max = E_SOI / d
mu_E = 1 - mu

v_circ = np.sqrt(mu_E / r_p)

a_max = 0.5 * (r_p + r_a_max)
v_max = np.sqrt(mu_E * (2 / r_p - 1 / a_max))



# Load initial conditions from data files
database = np.atleast_2d(np.loadtxt(DATA_FILE, skiprows=1))

SORT_BY = "E_SOI"
DESCENDING = True

columns = {"Cj": 6, "E_SOI": 7, "h_min_moon": 8, "h_last_earth": 9}


order = np.argsort(database[:, columns[SORT_BY]], kind="stable")
selection = database[order[::-1] if DESCENDING else order]

x0_selection = selection[:1, :6]


pool = get_context("fork").Pool()

sol = pool.starmap(integrate_one, [(x0, T_min, T_max, dt, earth_SOI_exit) for x0 in x0_selection])

T = [res[0] for res in sol]
X = [res[1] for res in sol]

X_bar = [Transformations.CR3BP_to_inertial(Transformations.CR3BP_normalized_units_to_SV(traj, d, G, M), n, time * TU) for time, traj in zip(T, X)]

pool.close()
pool.join()



# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")



# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Rotating frame
for traj in X:
    ax1.plot(traj[0] * d, traj[1] * d)
    ax1.scatter(traj[0, 0] * d, traj[1, 0] * d)

ax1.scatter(-mu*d, 0, color="blue", label="Earth")
ax1.scatter((1-mu)*d, 0, color="darkred", label="Moon")

ax1.set_title("Rotating Frame")
ax1.set_xlabel("X [km]")
ax1.set_ylabel("Y [km]")
ax1.set_aspect("equal")
ax1.set_box_aspect(1)
ax1.legend()


# Barycentric inertial frame
for time, traj in zip(T, X_bar):

    earth_I = Transformations.CR3BP_to_inertial(
        [-mu*d, 0, 0, 0, 0, 0],
        n,
        time * TU
    )

    moon_I = Transformations.CR3BP_to_inertial(
        [(1-mu)*d, 0, 0, 0, 0, 0],
        n,
        time * TU
    )

    ax2.plot(traj[0], traj[1])
    ax2.scatter(traj[0, 0], traj[1, 0])

    ax2.plot(earth_I[0], earth_I[1], "--", color="blue")
    ax2.plot(moon_I[0], moon_I[1], "--", color="darkred")

ax2.scatter(-mu*d, 0, color="blue", label="Earth at $t=0$")
ax2.scatter((1-mu)*d, 0, color="darkred", label="Moon at $t=0$")

ax2.set_title("Barycentric Inertial Frame")
ax2.set_xlabel("X [km]")
ax2.set_ylabel("Y [km]")
ax2.set_aspect("equal")
ax2.set_box_aspect(1)
ax2.legend()

plt.tight_layout()
plt.show()