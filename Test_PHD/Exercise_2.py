import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


mu_s = 1.3271244e11    # sun gravitational parameter (km^3/s^2)
mu_e = 398600          # earth gravitational parameter (km^3/s^2)


# Integration function
def TBP_ECI(t, x, mu):
    
    F = np.zeros(12)
    
    dx = x[0] - x[6]
    dy = x[1] - x[7]
    dz = x[2] - x[8]
    dist_cubed = (dx**2 + dy**2 + dz**2) ** 1.5
    
    F[0], F[1], F[2] = x[3], x[4], x[5]
    F[6], F[7], F[8] = x[9], x[10], x[11]

    F[3] = 0.
    F[4] = 0.
    F[5] = 0.
    
    F[9] =   mu * dx / dist_cubed
    F[10] =  mu * dy / dist_cubed
    F[11] =  mu * dz / dist_cubed
    
    return F



# Total energy function
def Total_energy(x):
    kinetic_energy, potential_energy = 0, 0
    
    for i in range(0, len(x), 6):
        vx, vy, vz = x[i+3], x[i+4],  x[i+5]
        kinetic_energy += 0.5  * (vx**2 + vy**2 + vz**2)
    
    for i in range(0, len(x), 6):
        for j in range(i + 6, len(x), 6):
            dx = x[i] - x[j]
            dy = x[i+1] - x[j+1]
            dz = x[i+2] - x[j+2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            potential_energy += - mu_e / r
            
    return kinetic_energy, potential_energy, kinetic_energy + potential_energy



# Total angular momentum function
def Total_angular_momentum(x):
    r1, v1 = np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]])   
    r2, v2 = np.array([x[6], x[7], x[8]]), np.array([x[9], x[10], x[11]])

    L1 = np.cross(r1, v1)
    L2 = np.cross(r2, v2)

    return np.linalg.norm(L1 + L2)



# Temporal array parameters
tmin = 0
tmax = 6052 * 2
dt = 10


# Initial conditions [x1, y1, z1,  vx1, vy1, vz1,    x2, y2, z2,  vx2, vy2, vz2]
x0 = [0., 0., 0.,  0., 0., 0.,  4604.49276873138, 1150.81472538679, 4694.55079634563,  -5.10903235110107, -2.48824074138143, 5.62098648967432]


# Define lists
E_K, E_P, E_tot = [], [], []
L = []


# Integration
T = np.arange(tmin, tmax, dt)
s = solve_ivp(TBP_ECI, [tmin, tmax], x0, args=(mu_e,), method='RK45', t_eval=T, rtol=1e-12, atol=1e-12)


# Initial energy and angular momentum
E0 = Total_energy(x0)
L0 = Total_angular_momentum(x0)


# Saving energy and angular momentum data
for i in range(len(T)):
    E_K.append(Total_energy(s.y[:, i])[0])
    E_P.append(Total_energy(s.y[:, i])[1])
    E_tot.append(Total_energy(s.y[:, i])[2])
    L.append(Total_angular_momentum(s.y[:,i]) - L0)


# Saving trajectories data
X1, Y1, Z1 = s.y[0], s.y[1], s.y[2]
X2, Y2, Z2 = s.y[6], s.y[7], s.y[8]



# Trajectory Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

RE = 6371.0
x0, y0, z0 = 0.0, 0.0, 0.0
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 80)
xs = x0 + RE*np.outer(np.cos(u), np.sin(v))
ys = y0 + RE*np.outer(np.sin(u), np.sin(v))
zs = z0 + RE*np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, color="tab:blue", alpha=0.8, linewidth=0, label="Earth", zorder=-1)

ax.plot3D(X1, Y1, Z1, label="Earth")
ax.plot3D(X2, Y2, Z2, label="Satellite", zorder=10)
ax.scatter3D(X1[0], Y1[0], Z1[0], s=50, zorder=-1)
ax.scatter3D(X2[0], Y2[0], Z2[0], s=50, zorder=10)

ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")

data_ranges = np.array([X1.max()-X1.min(), Y1.max()-Y1.min(), Z1.max()-Z1.min(), 2*RE, 2*RE, 2*RE])

R = data_ranges.max()/2
cx = (X1.max()+X1.min())/2; cy = (Y1.max()+Y1.min())/2; cz = (Z1.max()+Z1.min())/2
ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)

try:
    ax.set_box_aspect([1,1,1])  
except AttributeError:
    pass

plt.show()


# Energy plot
plt.figure(figsize=(12, 8))
plt.plot(T/3600, E_K, linewidth=2, label='Kinetic Energy')
plt.plot(T/3600, E_P, linewidth=2, label='Potential Energy')
plt.plot(T/3600, E_tot, linewidth=2, label='Total Energy')
plt.xlabel("Time [h]")
plt.ylabel("Energy [km^2/s^2]")
plt.legend()
plt.show()


# Angular momentum plot
plt.figure(figsize=(12, 8))
plt.plot(T/3600, L, linewidth=1)
plt.xlabel("Time [h]")
plt.ylabel("L - L0 [km^2/s]")
plt.show()