from multiprocessing import get_context

from numpy.linalg import norm
from Celestial_Mechanics import Transformations
from Celestial_Mechanics import Integration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import time


start_time = time.perf_counter()

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
LU = d



# Define Spheres of Influence
M_SOI = (m2/m1)**(2/5) * d
E_SOI = (m1/ms)**(2/5) * d_E 
SOI = ((m1 + m2)/ms)**(2/5) * (d_E + mu * d)


# Define Dimension Space
dim = '2D'
RETROGRADE = True
SAVE = True
PLOT = False



# Sample from sphere surface
def sample_on_ipersphere(R, dim, n_samples_theta, offset=None):

    if offset is None:
        offset = [0, 0, 0]

    theta = np.linspace(0, 2*np.pi, n_samples_theta, endpoint=False)

    if dim == '3D':

        cos_phi = np.random.uniform(-1, 1, n_samples_theta)

        sin_phi = np.sqrt(1 - cos_phi**2)

        x = R * np.cos(theta) * sin_phi + offset[0]
        y = R * np.sin(theta) * sin_phi + offset[1]
        z = R * cos_phi + offset[2]

        return np.column_stack((x.flatten(), y.flatten(), z))

    elif dim == '2D':

        x = R * np.cos(theta) + offset[0]
        y = R * np.sin(theta) + offset[1]
        z = np.full(x.size, offset[2])

        return np.column_stack((x, y, z))



# Define collision function
def min_distance(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Define moon SOI exit function
def moon_SOI(t, x, mu): 
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)
    return M_SOI / d - r_moon



# Define earth SOI exit function
def earth_SOI(t, x, mu): 
    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    return E_SOI / d - r_earth



# Define Lunar Radial Velocity
def lunar_radial_velocity(t, x, mu):

    r_dot = (x[0] - (1 - mu))*x[3] + x[1]*x[4] + x[2]*x[5]

    return r_dot



# Define 2 Body Energy
def two_body_energy(r, v, mu, body=2, frame='rotating'):

    r = np.asarray(r)
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0])
    off2 = np.array([1 - mu, 0, 0])

    offsets = [off1, off2]
    mu_bodies = [1 - mu, mu]

    offset = offsets[body - 1][:, None] if r.ndim == 2 else offsets[body - 1]
    r_rel = r - offset

    if frame == 'rotating':
        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r_rel, axis=0)
    else:
        v_rel = v

    return 0.5 * np.sum(v_rel**2, axis=0) - mu_bodies[body - 1] / np.linalg.norm(r_rel, axis=0)



# Integration function
def integrate_one(x0, T_min, T_max, dt, events=(min_distance, earth_SOI, moon_SOI)):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=1e-9,
        atol=1e-12,
        events=events
    )

    return sol





# Define integrator and event functions
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1

earth_SOI.terminal = True
earth_SOI.direction = -1

moon_SOI.direction = +1
moon_SOI.terminal = True



# Define Lists
times = []
SV = []
SV_inertial = []
moon_inertial = []
earth_inertial = []
moon_energy = []
earth_energy = []
Cj = []



# Define integration initial conditions
V_inf_min = 0 * TU/LU
V_inf_max = 3 * TU/LU



# Define integration quantities and counters
masses = [m1, m2]
T_min = 0
total = 0
forward_collisions = 0
retro_collisions = 0
n_samples_r = 100
n_samples_theta = 360


# Define integration time parameters
T_max = 2 * np.pi
dt = 0.01




# Create a multiprocessing pool using the "fork" context
pool = get_context("fork").Pool()



# Iterate over Jacobi constants
for v_inf in np.arange(V_inf_min, V_inf_max, 1):

    initial_conditions = []

    for theta in np.arange(0, 2 * np.pi, 1):

        x = (1 - mu) + M_SOI / d * np.cos(theta)
        y = M_SOI / d * np.sin(theta)
        z = 0

        for alpha in np.arange(0, 2 * np.pi, 0.1):

            vx = v_inf * np.cos(alpha)
            vy = v_inf * np.sin(alpha) + (1 - mu)
            vz = 0

            x0 = Transformations.inertial_to_CR3BP([x, y, z, vx, vy, vz], 1, 0)[:, 0]

            if lunar_radial_velocity(0, x0, mu) < 0:
                initial_conditions.append(x0)


    total += len(initial_conditions)
    initial_conditions = np.array(initial_conditions)



    # Integration
    solutions = pool.starmap(
                    integrate_one,
                        ((x0, T_min, T_max, dt, 
                            (min_distance)) 
                                for x0 in initial_conditions), chunksize=1)
    
    for sol in solutions:
    
        collided = len(sol[2][0]) > 0

        trajectory = Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M)

        moon_trajectory = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], n, sol[0]* TU)

        earth_trajectory = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], n, sol[0] * TU)

        bar_inertial = Transformations.CR3BP_to_inertial(trajectory, n, sol[0] * TU)

        e_energy = two_body_energy(sol[1][:3], sol[1][3:], mu, body=1, frame='rotating')

        m_energy = two_body_energy(sol[1][:3], sol[1][3:], mu, body=2, frame='rotating')

        Jacobi_constant = Integrator.CR3BP_Cj_from_v(sol[1], mu)       
    

        times.append(sol[0])
        SV.append(trajectory)
        SV_inertial.append(bar_inertial)
        moon_inertial.append(bar_inertial - moon_trajectory)
        earth_inertial.append(bar_inertial - earth_trajectory)
        moon_energy.append(m_energy)
        earth_energy.append(e_energy)
        Cj.append(Jacobi_constant)



pool.close()
pool.join()




# Compute Inertial Trajectories
T = np.linspace(0, 2 * np.pi, 10000)
Earth = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], 1, T)
Moon = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], 1, T)



# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")


# Plot the results
if dim=='3D':

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter3D(initial_positions[:,0] * d, initial_positions[:,1] * d, initial_positions[:,2] * d, s=0.5, alpha=0.8, color='grey')
    ax.scatter3D(initial_conditions[:, 0] * d, initial_conditions[:, 1] * d, initial_conditions[:, 2] * d, s=0.5, color='black', marker="o")

    for trajectory in SV:

        X = trajectory[0, :]
        Y = trajectory[1, :]
        Z = trajectory[2, :]

        ax.plot3D(X, Y, Z)

    ax.scatter3D(-mu*d, 0, 0, s=50, zorder=-1, label='Earth', color='b')
    ax.scatter3D((1-mu)*d, 0, 0, s=50, zorder=-1, label='Moon', color='r')

    ax.scatter3D(0.8369151258 * d, 0.0, 0.0, s=45, marker='x', color='black')
    ax.scatter3D(1.1556821654 * d, 0.0, 0.0, s=45, marker='x', color='black')
    #ax.scatter3D(-1.0050626458 * d, 0.0, 0.0, s=45, marker='x', color='grey')
    #ax.scatter3D((0.5 - mu) * d, np.sqrt(3) / 2 * d, 0.0, s=45, marker='x', color='grey')
    #ax.scatter3D((0.5 - mu) * d, -np.sqrt(3) / 2 * d, 0.0, s=45, marker='x', color='grey')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.legend()
    plt.show()


elif dim == '2D':

    fig = plt.figure(figsize=(19, 10))

    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[2, 1], hspace=0.2, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect(1)

    ax1.set_title("Rotating Frame")  

    ax1.scatter(initial_conditions[:, 0] * d, initial_conditions[:, 1] * d, s=0.5, color='black', marker="o")

    for trajectory in SV:

        X = trajectory[0, :]
        Y = trajectory[1, :]

        ax1.plot(X, Y)

    x = np.linspace(-1.5, 1.5, 600) 
    X, Y = np.meshgrid(x, x)

    r1 = np.sqrt((X + mu)**2 + Y**2)
    r2 = np.sqrt((X - (1 - mu))**2 + Y**2)
    C = X**2 + Y**2 + 2*((1 - mu)/r1 + mu/r2)


    ax1.scatter((1-mu)*d, 0, s=50, zorder=10, label='Moon', color='darkred')
    ax1.scatter(-mu*d, 0, s=50, zorder=10, label='Earth', color='b')
    ax1.scatter(0.8369151258 * d, 0, s=45, marker='x', color='black', label='Lagrange points', zorder=10)
    ax1.scatter(1.1556821654 * d, 0, s=45, marker='x', color='black', zorder=10)



    ax2.set_title("Baricentric Inertial Frame")  

    ax2.plot(Earth[0, :], Earth[1, :], '--', color='b', alpha=1, linewidth=2, zorder=10)
    ax2.plot(Moon[0, :], Moon[1, :], '--', color='darkred', alpha=1, linewidth=2, zorder=10)
    
    for traj in SV_inertial:

        ax2.plot(traj[0, :], traj[1, :], alpha=0.8)
        ax2.scatter(traj[0, 0], traj[1, 0], s=5, zorder=10)

    ax2.scatter(Earth[0, 0], Earth[1, 0], s=50, color='b', label='Earth', zorder=0)
    ax2.scatter(Moon[0, 0], Moon[1, 0], s=50, color='darkred', label='Moon', zorder=0) 



    ax3.set_title("Moon Inertial Frame")  

    ax3.scatter(0, 0, s=50, color='darkred', label='Moon', zorder=0) 
    
    for traj in moon_inertial:

        ax3.plot(traj[0, :], traj[1, :], alpha=0.8)
        ax3.scatter(traj[0, 0], traj[1, 0], s=5, zorder=10)


    
    for time, Cj_traj, e_energy, m_energy in zip(times, Cj, earth_energy, moon_energy):

        ax4.plot(time, Cj_traj - Cj_traj[0], alpha=0.8)
        ax5.plot(time, e_energy, alpha=0.8)
        ax6.plot(time, m_energy, alpha=0.8)

    ax4.axhline(0, color='k', linestyle='--', alpha=0.8)
    ax5.axhline(0, color='k', linestyle='--', alpha=0.8)
    ax6.axhline(0, color='k', linestyle='--', alpha=0.8)

    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True) 
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.set_xlabel("X [km]")
    ax1.set_ylabel("Y [km]")
    ax1.axis("equal")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True) 
    ax2.set_xlabel("X [km]") 
    ax2.axis("equal")
    ax2.grid(alpha=0.25) 
    ax2.legend()

    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax3.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True) 
    ax3.set_xlabel("X [km]") 
    ax3.axis("equal")
    ax3.grid(alpha=0.25) 
    ax3.legend()

    ax4.set_title("Jacobi Constant Error")
    ax4.set_xlabel("Time [TU]")
    ax4.set_ylabel(r"$C_J-C_J(0)$")

    ax5.set_title("Earth Specific Energy")
    ax5.set_xlabel("Time [TU]")
    ax5.set_ylabel(r"$\varepsilon_E$")

    ax6.set_title("Moon Specific Energy")
    ax6.set_xlabel("Time [TU]")
    ax6.set_ylabel(r"$\varepsilon_M$")

    plt.show()