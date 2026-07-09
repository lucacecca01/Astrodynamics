from Celestial_Mechanics import Transformations
from Celestial_Mechanics import Integration
import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
d = 384400       # Average distance between Earth and Moon in km
m1 = 5.974e24    # Mass of Earth in kg
m2 = 7.348e22    # Mass of Moon in kg
ms = 1.989e30    # Mass of Sun in kg
d_E = 1.496e8    # Average distance between Earth and Sun in km
M = m1 + m2      # Total mass of the Earth-Moon system in kg
mu = m2/M        # Moon-to-total mass ratio

M_SOI = d * (m2/m1)**(2/5)     # Sphere of influence of the Moon in km
E_SOI = d_E * (m1/ms)**(2/5)   # Sphere of influence of the Earth in km


initial_positions = []


# Define a sphere of radius E_SOI as initial position
for phi in np.arange(0, 2*np.pi, 0.1):
    for theta in np.arange(0, 2*np.pi, 0.1):
        x = E_SOI * np.cos(theta) * np.sin(phi) - mu*d
        y = E_SOI * np.sin(theta) * np.sin(phi)
        z = E_SOI * np.cos(phi)

        initial_positions.append([x,y,z])

initial_positions = np.array(initial_positions)




fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter3D(initial_positions[:,0], initial_positions[:,1], initial_positions[:,2], s=1, alpha=0.8, color='grey')

ax.scatter3D(-mu*d, 0, 0, s=50, zorder=-1, label='Earth', color='b')
ax.scatter3D((1-mu)*d, 0, 0, s=50, zorder=-1, label='Moon', color='r')

ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
ax.legend()
plt.show()

   
