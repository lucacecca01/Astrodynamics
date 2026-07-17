from multiprocessing import get_context
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
BYPASS = False



# Create Sphere function
def ipersphere(R, dim, offset=None):

    if offset is None:
            offset = [0, 0, 0]

    ipersphere = []

    if dim == '3D':

        for phi in np.arange(0, np.pi, 0.2):
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
def max_distance(t, x, mu, body='BAR'): 

    if body == 'BAR':
        r_bary = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        return SOI / d - r_bary
    
    elif body == 'EARTH':
        r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
        return E_SOI / d - r_earth
    
    elif body == 'MOON':
        r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)
        return M_SOI / d - r_moon



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



# Define Lunar Radial Velocity
def lunar_radial_velocity(r, v, mu):

    r_rel = np.asarray(r) - np.array([1 - mu, 0, 0])
    v_rel = np.asarray(v) + np.cross([0, 0, 1], r_rel)

    return np.dot(r_rel, v_rel) / np.linalg.norm(r_rel)



# Define Lunar Revolution Angle
def lunar_revolution_angle(t, x, mu):

    x_inertial = Transformations.CR3BP_to_inertial(x, 1, t)
    x_moon = Transformations.CR3BP_to_inertial([(1 - mu), 0, 0, 0, 0, 0], 1, t)

    x_moon_rel = x_inertial - x_moon

    rev_angle = np.unwrap(np.arctan2(x_moon_rel[1], x_moon_rel[0]))

    return rev_angle



# Define Moon Capture Function
def moon_capture(t, x, mu):

    return two_body_energy(x[:3], x[3:], mu, body=2, frame='rotating')



# Define the initial conditions filters
def accept_initial_condition(pos, v, mu, filters, sol=None, t=None, x=None):

    conditions = []

    if filters["ballistic capture_start"]:
        conditions.append(two_body_energy(pos, v, mu, body=2, frame='rotating') > 0)
        conditions.append(two_body_energy(pos, v, mu, body=1, frame='rotating') < 0)
        conditions.append(two_body_energy_derivative(pos, v, mu, body=2, frame='rotating') < 0)

    if filters["capture"]:
        r_M = np.sqrt((sol[1][0] - (1 - mu))**2 + sol[1][1]**2 + sol[1][2]**2)
        conditions.append(np.all(r_M < 2 * M_SOI / d))
     
    if filters["ballistic capture_end"]:
        conditions.append(two_body_energy(pos, v, mu, body=2, frame='rotating') < 0)
        conditions.append(two_body_energy_derivative(pos, v, mu, body=2, frame='rotating') > 0)

    if filters["revolution_angle"]:
        theta = lunar_revolution_angle(sol[0], sol[1], mu)
        conditions.append(max(abs(theta - theta[0])) > revolution_number * 2 * np.pi)
  
    if filters["no_collision"]:
        conditions.append(len(sol[2][0]) == 0)

    if filters["no_SOI_exit"]:
        conditions.append(len(sol[2][1]) == 0)

    if filters["no_escape"]:
        theta = lunar_revolution_angle(sol[0], sol[1], mu)
        conditions.append(len(sol[2][2]) == 0 or max(abs(theta - theta[0])) > 2 * np.pi)
        
    return all(conditions)



# Integration function
def integrate_one(x0):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=1e-9,
        atol=1e-12,
        events=(min_distance, max_distance, moon_capture)
    )

    return sol



# Define integrator
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1

max_distance.terminal = True
max_distance.direction = -1

moon_capture.terminal = True
moon_capture.direction = 1



# Conditions to apply
filters_start = {
    "ballistic capture_start": True,
    "capture": False,
    "ballistic capture_end": False,
    "revolution_angle": False,
    "no_collision": False,
    "no_SOI_exit": False,
    "no_escape": False
}


filters_end = {
    "ballistic capture_start": False,
    "capture": True,
    "ballistic capture_end": True,
    "revolution_angle": True,
    "no_collision": True,
    "no_SOI_exit": True,
    "no_escape": True
}



# Define Lists
initial_conditions = []
SV = []
SV_inertial = []
moon_inertial = []
earth_inertial = []



# Define integration initial conditions
Cj_max = 3.19
Cj_min = 2.96
Cj_norm = 3.05



# Define integration boundaries
masses = [m1, m2]
T_max = 2*np.pi
T_min = 0
dt = 0.001
total = 0
collisions = 0
SOI_exits = 0
uncaptures = 0
captured = 0
max_revolutions = 0
n_sample = 1000
revolution_number = 2



#Define a sphere of radius M_SOI as initial position                        
initial_positions = ipersphere((1 - 1e-12) * M_SOI / d, dim, [(1-mu), 0, 0])



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



# Integration in normalized units and conversion into physical units
with get_context("fork").Pool() as pool:
    results = pool.map(integrate_one, initial_conditions)

for sol in results:

    theta = lunar_revolution_angle(sol[0], sol[1], mu)
    r_M = np.sqrt((sol[1][0] - (1 - mu))**2 + sol[1][1]**2 + sol[1][2]**2)

    if len(sol[2][0]) > 0:
        collisions += 1

    if len(sol[2][1]) > 0:
        SOI_exits += 1

    if len(sol[2][2]) > 0 and max(abs(theta - theta[0])) < 2 * np.pi:
        uncaptures += 1

    if two_body_energy(sol[1][:3, -1], sol[1][3:, -1], mu, body=2, frame='rotating') < 0 and len(sol[2][1]) == 0 and len(sol[2][0]) == 0 and np.all(r_M < 2 * M_SOI / d):

        if len(sol[2][2]) == 0 or max(abs(theta - theta[0])) > 2 * np.pi:
            captured += 1

        if max(abs(theta - theta[0])) > revolution_number * 2 * np.pi:
            max_revolutions += 1
    

    if (accept_initial_condition(sol[1][:3, -1], sol[1][3:, -1], mu, filters_end, sol) or BYPASS):

        trajectory = Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M)

        moon_trajectory = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], n, sol[0]* TU)

        earth_trajectory = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], n, sol[0] * TU)

        bar_inertial = Transformations.CR3BP_to_inertial(trajectory, n, sol[0] * TU)

        SV.append(trajectory)
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
print(f"Collisions:                       {collisions} ({100 * collisions / total:.1f} %)")
print(f"BAR_SOI Exits:                    {SOI_exits} ({100 * SOI_exits / total:.1f} %)")
print(f"Escaped with N_rev < 1:           {uncaptures} ({100 * uncaptures / total:.1f} %)")
print(f"Still bounded or N_rev >= 1:      {captured} ({100 * (captured) / total:.1f} %)")
print(f"N_rev >= {revolution_number}:                       {max_revolutions} ({100 * (max_revolutions) / total:.1f} %)")



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
        ax2.scatter(traj[0, 0], traj[1, 0], s=5, zorder=10)

    ax2.scatter(Earth[0, 0], Earth[1, 0], s=50, color='b', label='Earth', zorder=0)
    ax2.scatter(Moon[0, 0], Moon[1, 0], s=50, color='darkred', label='Moon', zorder=0) 



    ax3.set_title("Moon Inertial Frame")  

    ax3.scatter(0, 0, s=50, color='darkred', label='Moon', zorder=0) 
    
    for traj in moon_inertial:

        ax3.plot(traj[0, :], traj[1, :], alpha=0.8)
        ax3.scatter(traj[0, 0], traj[1, 0], s=5, zorder=10)

    
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