from Celestial_Mechanics import Transformations
from Celestial_Mechanics import Integration
import numpy as np
import matplotlib.pyplot as plt

# Define constants
G = 6.67430e-20          # Gravitational constant in km^3 kg^-1 s^-2
d = 384400               # Average distance between Earth and Moon in km
m1 = 5.974e24            # Mass of Earth in kg
m2 = 7.348e22            # Mass of Moon in kg
ms = 1.989e30            # Mass of Sun in kg
d_E = 1.496e8            # Average distance between Earth and Sun in km
R_E = 6378               # Earth radius in km
R_L = 1737               # Moon radius in km
day = 86400              # Day in s
M = m1 + m2              # Total mass of the Earth-Moon system in kg
mu = m2/M                # Moon-to-total mass ratio



# Define Spheres of Influence
M_SOI = (m2/m1)**(2/5) * d
E_SOI = (m1/ms)**(2/5) * d_E 
SOI = ((m1 + m2)/ms)**(2/5) * (d_E + mu * d)



# Define Dimension Space
dim = '2D'



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
def two_body_energy(r, v, mu, body=2):

    r = np.asarray(r)
    v = np.asarray(v)

    r1 = np.linalg.norm(r - np.array([-mu, 0, 0]))
    r2 = np.linalg.norm(r - np.array([1 - mu, 0, 0]))

    distances = [r1, r2]
    mu_bodies = [1 - mu, mu]

    return 0.5 * np.dot(v, v) - mu_bodies[body - 1] / distances[body - 1]




# Define 2 Body Energy Derivative
def two_body_energy_derivative(r, v, mu, body=2):

    r = np.asarray(r) 
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0])
    off2 = np.array([1 - mu, 0, 0])

    r1 = np.linalg.norm(r - off1)
    r2 = np.linalg.norm(r - off2)

    ax = 2 * v[1]  +  r[0]  -  (1 - mu) * (r[0] + mu) / r1**3  -  mu * (r[0] - (1 - mu)) / r2**3
    ay = -2 * v[0]  +  r[1]  -  (1 - mu) * r[1] / r1**3 - mu * r[1] / r2**3
    az = - (1 - mu) * r[2] / r1**3 - mu * r[2] / r2**3

    a = np.array([ax, ay, az])

    distances = [r1, r2]
    offset = [off1, off2]
    mu_bodies = [1 - mu, mu]

    return np.dot(v, a) + mu_bodies[body - 1] / distances[body - 1]**3 * np.dot(r - offset[body - 1], v)





# Define integrator
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1

max_distance.terminal = True
max_distance.direction = -1



# Define Lists
initial_conditions = []
SV = []



# Define integration initial conditions
x0_norm = [9.0453898750573813E-1, 0, 1.4388186844294218E-1,    0, -4.9801575824700677E-2, 0]
Cj_max = 3.19
Cj_min = 2.96

Cj_norm = 3.05



# Define integration boundaries
masses = [m1, m2]
T_max = 5
T_min = 0
dt = 0.001
collisions = 0
escapes = 0
n_sample = 10



#Define a sphere of radius E_SOI as initial position                        
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

        if two_body_energy_derivative(pos, v, mu) <= 0 and two_body_energy(pos, v, mu) < 0:
            initial_conditions.append(np.concatenate((pos, v)))



# Initialize initial conditions
initial_conditions = np.array(initial_conditions)
#initial_conditions = np.array([x0_norm])



# Integration in normalized units and conversion into physical units
for i, x0 in enumerate (initial_conditions):

    sol = Integrator.Integrator(masses, G, x0, T_min, T_max, dt, model='CR3BP', integrator='scipy', method='RK45', rtol=1e-9, atol=1e-12, events=(min_distance, max_distance))

    if len(sol[2][0]) > 0:
        collisions += 1

    if len(sol[2][1]) > 0:
        escapes += 1

    SV.append(Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M))



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

    plt.figure(figsize=(10,10))

    plt.scatter(initial_positions[:,0] * d, initial_positions[:,1] * d, s=0.5, alpha=0.8, color='grey')
    plt.scatter(initial_conditions[:, 0] * d, initial_conditions[:, 1] * d, s=0.5, color='black', marker="o")

    for trajectory in SV:

        X = trajectory[0, :]
        Y = trajectory[1, :]

        plt.plot(X, Y)


    x = np.linspace(-1.5, 1.5, 600) 
    X, Y = np.meshgrid(x, x)

    r1 = np.sqrt((X + mu)**2 + Y**2)
    r2 = np.sqrt((X - (1 - mu))**2 + Y**2)
    C = X**2 + Y**2 + 2*((1 - mu)/r1 + mu/r2)

    plt.contour(X*d, Y*d, C, levels=[Cj_norm], colors="k")

    plt.scatter((1-mu)*d, 0, s=50, zorder=10, label='Moon', color='darkred')
    plt.scatter(-mu*d, 0, s=50, zorder=10, label='Earth', color='b')
    plt.scatter(0.8369151258 * d, 0, s=45, marker='x', color='black', label='Lagrange points')
    plt.scatter(1.1556821654 * d, 0, s=45, marker='x', color='black')

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")

    plt.legend()
    plt.show()

   
