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



# Create Sphere function
def sphere(R, offset=None):

    if offset is None:
        offset = [0, 0, 0]

    sphere = []

    for phi in np.arange(0, 2*np.pi, 0.2):
        for theta in np.arange(0, 2*np.pi, 0.2):

            x = R * np.cos(theta) * np.sin(phi) + offset[0]
            y = R * np.sin(theta) * np.sin(phi) + offset[1]
            z = R * np.cos(phi) + offset[2]

            sphere.append([x,y,z])

    return np.array(sphere)



# Sample from sphere surface
def sample_on_sphere(radius, n_samples):

    points = np.random.normal(size=(n_samples, 3))

    points /= np.linalg.norm(points, axis=1, keepdims=True)

    return radius * points



# Define collision function
def min_distance(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Define integrator
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1



# Define Lists
initial_conditions = []
SV = []



# Define integration initial conditions
x0_norm = [1.0664483173291939E+0, 0, 0,    0, 3.0863475065250201E-1, 0]
Cj_norm = 3



# Define integration boundaries
masses = [m1, m2]
T_max = 0.5
T_min = 0
dt = 0.01
collisions = 0



#Define a sphere of radius E_SOI as initial position
M_SOI = d * (m2/m1)**(2/5)                                # Sphere of influence of the Moon in km
E_SOI = d_E * (m1/ms)**(2/5)                              # Sphere of influence of the Earth in km
initial_positions = sphere(M_SOI / d, [(1-mu), 0, 0])



# Define initial velocities
for pos in initial_positions:
    v_norm = Integrator.CR3BP_v_from_Cj(Cj_norm, pos, mu)
    v_set = sample_on_sphere(v_norm, 10)

    for v in v_set:
       initial_conditions.append(np.concatenate((pos, v)))



# Initialize initial conditions
#initial_conditions = np.array(initial_conditions)
initial_conditions = np.array([x0_norm])



# Integration in normalized units and conversion into physical units
for i, x0 in enumerate (initial_conditions):

    sol = Integrator.Integrator(masses, G, x0, T_min, T_max, dt, model='CR3BP', integrator='scipy', method='RK45', rtol=1e-9, atol=1e-12, events=min_distance)

    if sol[1].shape[1] < (int(round((T_max - T_min) / dt)) + 1):
        collisions += 1

    SV.append(Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M))



# Print results
print(collisions)




# Plot the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter3D(initial_positions[:,0] * d, initial_positions[:,1] * d, initial_positions[:,2] * d, s=0.5, alpha=0.8, color='grey')
ax.scatter3D(initial_conditions[:, 0] * d, initial_conditions[:, 1] * d, initial_conditions[:, 2] * d, s=0.5, color='black', marker="o")

for trajectory in SV:

    X = trajectory[0, :]
    Y = trajectory[1, :]
    Z = trajectory[2, :]

    ax.plot3D(X, Y, Z)

#ax.scatter3D(-mu*d, 0, 0, s=50, zorder=-1, label='Earth', color='b')
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

   
