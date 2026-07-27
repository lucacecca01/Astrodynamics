from Celestial_Mechanics import Integration, Transformations
import numpy as np
from multiprocessing import get_context
import time
from matplotlib import pyplot as plt



# Define integration function
def integrate_one(x0, T_min, T_max, dt):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='NBP',
        integrator='rebound',
        method='ias15',
        rtol=1e-9,
        stop_condition=earth_SOI_exit,
    )

    return sol



# Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, state):
    earth = state[0]
    spacecraft = state[2]

    r = spacecraft[:3] - earth[:3]
    v = spacecraft[3:] - earth[3:]

    return np.linalg.norm(r) >= E_SOI and np.dot(r, v) > 0


   
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


# Input
DATA_FILES = {
    "escape": "Escape_trajectories.txt",
    "capture": "Capture_trajectories.txt",
}



# Initialize the integrator
start_time = time.perf_counter()
Integrator = Integration()



# Initialize lists to store results
T = []
X = []
x0 = []



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0
T_max = 10 * 2*np.pi * TU
dt = 0.001 * TU



# Load initial conditions from data files
escape_database, capture_database = [np.atleast_2d(np.genfromtxt(filename, skip_header=1)) for filename in DATA_FILES.values()]



# Sort the databases by a specific column (e.g., "DV") in ascending or descending order
SORT_BY = "DV"
DESCENDING = True

columns = {"Cj": 7, "DE": 8, "DV": 9, "Emax": 10,
           "h_perigeo": 11, "h_periluneo": 12}

escape_performance = escape_database[:, columns["Emax"]] / escape_database[:, columns["DV"]]**2
capture_performance = capture_database[:, columns["Emax"]] / capture_database[:, columns["DV"]]**2

order = np.argsort(escape_performance, kind="stable")
escape_database = escape_database[order[::-1] if DESCENDING else order]

order = np.argsort(capture_performance, kind="stable")
capture_database = capture_database[order[::-1] if DESCENDING else order]



# Extract initial conditions and times from the databases
escape_initial_conditions = escape_database[:, :6]
escape_initial_times = escape_database[:, 6]
capture_initial_conditions = capture_database[:, :6]
capture_initial_times = capture_database[:, 6]




# Use multiprocessing to integrate trajectories in parallel
pool = get_context("fork").Pool()

for x0_traj, epoch, row in zip(escape_initial_conditions[0:5], escape_initial_times[0:5], escape_database[0:5]):
    print(f"DE: {row[columns['DE']]:.3f} | Emax: {row[columns['Emax']]:.3f} | DV: {row[columns['DV']]:.3f} | hmax: {row[columns['h_perigeo']]:.0f} km | hmin: {row[columns['h_periluneo']]:.0f} km | Cj: {row[columns['Cj']]:.3f}")
    earth = Transformations.CR3BP_to_inertial([-mu*d, 0, 0, 0, 0, 0], n, epoch)[:, 0]
    moon = Transformations.CR3BP_to_inertial([(1-mu)*d, 0, 0, 0, 0, 0], n, epoch)[:, 0]
    x0.append(np.concatenate((earth, moon, x0_traj)))


sol = pool.starmap(integrate_one, [(state, T_min, T_max, dt) for state in x0])

T = [sol[0] for sol in sol]
X = [sol[1] for sol in sol]

pool.close()
pool.join()



# Plotting the results
plt.figure(figsize=(10, 10))

for traj in X:
    earth = traj[0, :, :]
    moon = traj[1, :, :]
    spacecraft = traj[2, :, :]

    Vf = np.linalg.norm(spacecraft[-1, 3:]) 
    a = (2 / d_E - (Vf + 29.78)**2 / G / ms)**(-1)
    r_a = 2*a - d_E
    print(f"Final velocity: {Vf:.3f} km/s  ({Vf+29.78:.3f} km/s with apohelium of {r_a/d_E:.3f} AU)")

    plt.plot(earth[:, 0], earth[:, 1], label='Earth', color='blue')
    plt.plot(moon[:, 0], moon[:, 1], label='Moon', color='darkred')
    plt.plot(spacecraft[:, 0], spacecraft[:, 1])


plt.axis("equal")
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.title('Escape Trajectories')
plt.show()



# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")