from Celestial_Mechanics import Transformations
from Celestial_Mechanics import Integration
import numpy as np
import matplotlib.pyplot as plt

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



# Define Spheres of Influence
M_SOI = (m2/m1)**(2/5) * d
E_SOI = (m1/ms)**(2/5) * d_E 
SOI = ((m1 + m2)/ms)**(2/5) * (d_E + mu * d)



# Define Dimension Space
dim = '2D'
BYPASS = True



# Create Sphere function
def ipersphere(R, dim, offset=None):

    if offset is None:
            offset = [0, 0, 0]

    ipersphere = []

    if dim == '3D':

        for phi in np.arange(0, 2*np.pi, 0.2):
            for theta in np.arange(0, 2*np.pi, 0.2):

                x = R * np.cos(theta) * np.sin(phi) + offset[0]
                y = R * np.sin(theta) * np.sin(phi) + offset[1]
                z = R * np.cos(phi) + offset[2]

                ipersphere.append([x,y,z])


    elif dim == '2D':

        for theta in np.arange(0, 2*np.pi, 0.2):

            x = R * np.cos(theta) + offset[0]
            y = R * np.sin(theta) + offset[1]
            z = offset[2]

            ipersphere.append([x,y,z])


    return np.array(ipersphere)



# Sample from sphere surface
def sample_on_ipersphere(R, dim, n_samples):

    theta = np.random.uniform(0, 2*np.pi, n_samples)

    if dim == '3D':

        cos_phi = np.random.uniform(-1, 1, n_samples)
        sin_phi = np.sqrt(1 - cos_phi**2)

        vx = R * np.cos(theta) * sin_phi
        vy = R * np.sin(theta) * sin_phi
        vz = R * cos_phi

        return np.column_stack((vx, vy, vz))

    elif dim == '2D':
        vx = R * np.cos(theta) 
        vy = R * np.sin(theta)

        return np.column_stack((vx, vy))



# Define collision function
def min_distance(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Define SOI exit function
def max_distance(t, x, mu):   

    r_bary = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

    return SOI / d - r_bary



# Define 2 Body Energy
def two_body_energy(r, v, mu, body=2, frame='rotating'):

    r = np.asarray(r)
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0])
    off2 = np.array([1 - mu, 0, 0])

    offsets = [off1, off2]
    mu_bodies = [1 - mu, mu]

    r_rel = r - offsets[body - 1]

    if frame == 'rotating':
        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r_rel)
    else:
        v_rel = v

    return 0.5 * np.dot(v_rel, v_rel) - mu_bodies[body - 1] / np.linalg.norm(r_rel)



# Define 2 Body Energy Derivative
def two_body_energy_derivative(r, v, mu, body=2, frame='rotating'):

    r = np.asarray(r) 
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0])
    off2 = np.array([1 - mu, 0, 0])

    offsets = [off1, off2]
    mu_bodies = [1 - mu, mu]

    r_rel = r - offsets[body - 1]
    mu_body = mu_bodies[body - 1]

    r1 = np.linalg.norm(r - off1)
    r2 = np.linalg.norm(r - off2)

    ax = 2 * v[1]  +  r[0]  -  (1 - mu) * (r[0] + mu) / r1**3  -  mu * (r[0] - (1 - mu)) / r2**3
    ay = -2 * v[0]  +  r[1]  -  (1 - mu) * r[1] / r1**3 - mu * r[1] / r2**3
    az = - (1 - mu) * r[2] / r1**3 - mu * r[2] / r2**3

    a_rot = np.array([ax, ay, az])

    if frame == 'rotating':

        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r_rel)

        a_rel = (a_rot + 2 * np.cross(omega, v) + np.cross(omega, np.cross(omega, r)))
        a_body = np.cross(omega, np.cross(omega, offsets[body - 1]))

        a_rel = a_rel - a_body

    else:
        v_rel = v
        a_rel = a_rot


    return np.dot(v_rel, a_rel) + mu_body / np.linalg.norm(r_rel)**3 * np.dot(r_rel, v_rel)



# Define the initial conditions filters
def accept_initial_condition(pos, v, mu, filters):

    conditions = []

    if filters["negative_earth_energy"]:
        conditions.append(two_body_energy(pos, v, mu, body=1, frame='rotating') < 0)

    if filters["positive_earth_energy"]:
        conditions.append(two_body_energy(pos, v, mu, body=1, frame='rotating') > 0)

    if filters["negative_moon_energy"]:
        conditions.append(two_body_energy(pos, v, mu, body=2, frame='rotating') < 0)

    if filters["positive_moon_energy"]:
        conditions.append(two_body_energy(pos, v, mu, body=2, frame='rotating') > 0)

    if filters["decreasing_earth_energy"]:
        conditions.append(two_body_energy_derivative(pos, v, mu, body=1, frame='rotating') < 0)

    if filters["decreasing_moon_energy"]:
        conditions.append(two_body_energy_derivative(pos, v, mu, body=2, frame='rotating') < 0)

    if filters["increasing_earth_energy"]:
        conditions.append(two_body_energy_derivative(pos, v, mu, body=1, frame='rotating') > 0)

    if filters["increasing_moon_energy"]:
        conditions.append(two_body_energy_derivative(pos, v, mu, body=2, frame='rotating') > 0)

    if filters["no_collision"]:
        conditions.append(len(sol[2][0]) == 0)

    if filters["no_escape"]:
        conditions.append(len(sol[2][1]) == 0)

    return all(conditions)


    global _WORKER_CONFIG
    global _WORKER_INTEGRATOR

    _WORKER_CONFIG = config
    _WORKER_INTEGRATOR = Integration()




# Define integrator
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1

max_distance.terminal = True
max_distance.direction = -1



# Conditions to apply
filters_start = {
    "negative_earth_energy": False,
    "negative_moon_energy": False,
    "positive_earth_energy": False,
    "positive_moon_energy": True,
    "increasing_earth_energy": False,
    "decreasing_earth_energy": False,
    "increasing_moon_energy": False,
    "decreasing_moon_energy": True,
    "no_collision": False,
    "no_escape": False
}


filters_end = {
    "negative_earth_energy": False,
    "negative_moon_energy": True,
    "positive_earth_energy": False,
    "positive_moon_energy": False,
    "increasing_earth_energy": False,
    "decreasing_earth_energy": False,
    "increasing_moon_energy": True,
    "decreasing_moon_energy": False,
    "no_collision": True,
    "no_escape": False
}



# Define Lists
initial_conditions = []
SV = []
SV_inertial = []
moon_inertial = []
earth_inertial = []



# Define integration initial conditions
x0_norm = [9.0453898750573813E-1, 0, 1.4388186844294218E-1,    0, -4.9801575824700677E-2, 0]
Cj_max = 3.19
Cj_min = 2.96

Cj_norm = 3.05



# Define integration boundaries
masses = [m1, m2]
T_max = 2*np.pi
T_min = 0
dt = 0.01
collisions = 0
escapes = 0
n_sample = 100



#Define a sphere of radius M_SOI as initial position                        
initial_positions = ipersphere(M_SOI / d, dim, [(1-mu), 0, 0])



# Define initial velocities
for pos in initial_positions:
    try:
        v_norm = Integrator.CR3BP_v_from_Cj(Cj_norm, pos, mu)
    except ValueError: 
        continue

    v_set = sample_on_ipersphere(v_norm, dim, n_sample)


    for v in v_set:

        if dim == '2D':
            v = np.concatenate((v, [0]))

        if accept_initial_condition(pos, v, mu, filters_start) or BYPASS:
            initial_conditions.append(np.concatenate((pos, v)))



# Initialize initial conditions
initial_conditions = np.array(initial_conditions)
#initial_conditions = np.array([x0_norm])



# Integration in normalized units and conversion into physical units
for i, x0 in enumerate (initial_conditions):

    sol = Integrator.Integrator(masses, G, x0, T_min, T_max, dt, model='CR3BP', integrator='scipy', method='DOP853', rtol=1e-9, atol=1e-12, events=(min_distance, max_distance))

    if accept_initial_condition(sol[1][:3,-1], sol[1][3:,-1], mu, filters_end) or BYPASS:

        if len(sol[2][0]) > 0:
            collisions += 1

        if len(sol[2][1]) > 0:
            escapes += 1

        trajectory = Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M)
        SV.append(trajectory)

        moon_trajectory = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], n, sol[0] * TU)
        earth_trajectory = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], n, sol[0] * TU)

        bar_inertial = Transformations.CR3BP_to_inertial(trajectory, n, sol[0] * TU)
        SV_inertial.append(bar_inertial)

        moon_inertial.append(bar_inertial - moon_trajectory)
        earth_inertial.append(bar_inertial - earth_trajectory)



# Compute Inertial Trajectories
T = np.linspace(T_min, T_max, 10000)
Earth = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], 1, T)
Moon = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], 1, T)



# Print results
total = len(initial_conditions)
print("\n=== Simulation Results ===")
print(f"Total trajectories: {total}")
print(f"Collisions:         {collisions} ({100 * collisions / total:.1f} %)")
print(f"Escapes:            {escapes} ({100 * escapes / total:.1f} %)")



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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))


    ax1.set_title("Rotating Frame")  

    ax1.scatter(initial_positions[:,0] * d, initial_positions[:,1] * d, s=0.5, alpha=0.8, color='grey')
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

    ax1.contour(X*d, Y*d, C, levels=[Cj_norm], colors="k")

    ax1.scatter((1-mu)*d, 0, s=50, zorder=10, label='Moon', color='darkred')
    ax1.scatter(-mu*d, 0, s=50, zorder=10, label='Earth', color='b')
    ax1.scatter(0.8369151258 * d, 0, s=45, marker='x', color='black', label='Lagrange points', zorder=10)
    ax1.scatter(1.1556821654 * d, 0, s=45, marker='x', color='black', zorder=10)



    ax2.set_title("Baricentric Inertial Frame")  

    ax2.plot(Earth[0, :], Earth[1, :], '--', color='b', alpha=1, linewidth=2, zorder=10)
    ax2.plot(Moon[0, :], Moon[1, :], '--', color='darkred', alpha=1, linewidth=2, zorder=10)
    
    for traj in SV_inertial:

        ax2.plot(traj[0, :], traj[1, :], alpha=0.8)

    ax2.scatter(Earth[0, 0], Earth[1, 0], s=50, color='b', label='Earth', zorder=0)
    ax2.scatter(Moon[0, 0], Moon[1, 0], s=50, color='darkred', label='Moon', zorder=0) 



    ax3.set_title("Moon Inertial Frame")  

    ax3.scatter(0, 0, s=50, color='darkred', label='Moon', zorder=0) 
    
    for traj in moon_inertial:

        ax3.plot(traj[0, :], traj[1, :], alpha=0.8)


    
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

    plt.show()

   
