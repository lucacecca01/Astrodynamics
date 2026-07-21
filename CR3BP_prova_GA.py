from multiprocessing import get_context
from Celestial_Mechanics import Transformations
from Celestial_Mechanics import Integration
import numpy as np
import matplotlib.pyplot as plt
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



# Define Spheres of Influence
M_SOI = (m2/m1)**(2/5) * d
E_SOI = (m1/ms)**(2/5) * d_E 
SOI = ((m1 + m2)/ms)**(2/5) * (d_E + mu * d)



# Define Dimension Space
dim = '2D'
BYPASS = False
RETROGRADE = True



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

    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

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
def max_distance(t, x, mu, body='MOON'): 

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

    offset = offsets[body - 1][:, None] if r.ndim == 2 else offsets[body - 1]
    r_rel = r - offset

    if frame == 'rotating':
        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r_rel, axis=0)
    else:
        v_rel = v

    return 0.5 * np.sum(v_rel**2, axis=0) - mu_bodies[body - 1] / np.linalg.norm(r_rel, axis=0)



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



# Define Earth Revolution Angle
def earth_revolution_angle(t, x, mu):

    x_inertial = Transformations.CR3BP_to_inertial(x, 1, t)
    x_earth = Transformations.CR3BP_to_inertial([-mu, 0, 0, 0, 0, 0], 1, t)

    x_earth_rel = x_inertial - x_earth

    rev_angle = np.unwrap(np.arctan2(x_earth_rel[1], x_earth_rel[0]))

    return rev_angle



# Define Moon Capture Function
def moon_capture(t, x, mu):

    return two_body_energy(x[:3], x[3:], mu, body=2, frame='rotating')



# Define the initial conditions filters
def accept_initial_condition(pos, v, mu, filters, sol=None, SOI_exit_index=None):

    conditions = []

    if filters["escape"]:
        conditions.append(two_body_energy(pos, v, mu, body=1, frame='rotating') < 0)

    if filters["capture"]:
        conditions.append(two_body_energy(pos, v, mu, body=1, frame='rotating') > 0)

    if filters["lunar_flyby"]:
        conditions.append(two_body_energy(pos, v, mu, body=2, frame='rotating') > 0)
        conditions.append(lunar_radial_velocity(pos, v, mu) < 0)
   
    return all(conditions)



# Define the final conditions filters
def accept_final_condition(pos, v, mu, filters, sol=None, SOI_exit_index=None, rev_n=1):

    conditions = []
    moon_earth_energy = two_body_energy([1 - mu, 0, 0], [0, 0, 0], mu, body=1, frame='rotating')

    if filters["escape"]:
        conditions.append(two_body_energy(pos[:, -1], v[:, -1], mu, body=1, frame='rotating') > 0)


    if filters["capture"]:
        conditions.append(two_body_energy(pos[:, -1], v[:, -1], mu, body=1, frame='rotating') < moon_earth_energy)


    if filters["lunar_flyby"]:
        if SOI_exit_index is None:
            return False
        
        moon_energy_SOI_exit = two_body_energy(pos[:, SOI_exit_index], v[:, SOI_exit_index], mu, body=2, frame='rotating')
        moon_final_distance = np.linalg.norm(pos[:,-1] - np.array([(1 - mu), 0, 0]))

        conditions.append(moon_final_distance > M_SOI / d)
        conditions.append(moon_energy_SOI_exit > 0)


    if filters["maximum_revolutions"]:
        if SOI_exit_index is None:
            return False

        theta_earth_SOI_exit = earth_revolution_angle(sol[0][:SOI_exit_index + 1], sol[1][:, :SOI_exit_index + 1], mu)
        conditions.append(max(abs(theta_earth_SOI_exit - theta_earth_SOI_exit[0])) < rev_n * 2*np.pi)


    if filters["no_collision"]:
        conditions.append(len(sol[2][0]) == 0)

    return all(conditions)



# Integration function
def integrate_one(x0, T_min, T_max, dt):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=1e-9,
        atol=1e-12,
        events=(min_distance, max_distance)
    )

    trajectory = Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M)

    moon_trajectory = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], n, sol[0]* TU)

    earth_trajectory = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], n, sol[0] * TU)

    bar_inertial = Transformations.CR3BP_to_inertial(trajectory, n, sol[0] * TU)

    e_energy = two_body_energy(sol[1][:3], sol[1][3:], mu, body=1, frame='rotating')

    m_energy = two_body_energy(sol[1][:3], sol[1][3:], mu, body=2, frame='rotating')

    Jacobi_constant = Integrator.CR3BP_Cj_from_v(sol[1], mu)

    return sol, trajectory, bar_inertial, moon_trajectory, earth_trajectory, e_energy, m_energy, Jacobi_constant



# Define integrator
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1

max_distance.terminal = False
max_distance.direction = -1



# Conditions to apply
filters = {
    "escape": False,
    "capture": True,
    "lunar_flyby": True,
    "no_collision": True,
    "maximum_revolutions": True
}




# Define Lists
initial_conditions = []
times = []
SV = []
SV_inertial = []
moon_inertial = []
earth_inertial = []
moon_energy = []
earth_energy = []
Cj = []
x0_retro = []
forward_results = []
retro_results = []


# Define integration initial conditions
Cj_max = 3.19
Cj_min = 2.96
Cj_norm = 3.05



# Define integration boundaries
masses = [m1, m2]
T_max = 2*np.pi
T_min = 0
dt = 0.01
T_retro = -0.5*np.pi
dt_retro = -0.01
total = 0
collisions = 0
n_sample = 100
rev_n = 2



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

        if accept_initial_condition(pos, v, mu, filters) or BYPASS:
            initial_conditions.append(np.concatenate((pos, v)))



# Initialize initial conditions
initial_conditions = np.array(initial_conditions)



# Integration in normalized units and conversion into physical units
with get_context("fork").Pool() as pool:
    results = pool.starmap(integrate_one, ((x0, T_min, T_max, dt) for x0 in initial_conditions))


# Fill the lists with the accepted trajectories
for result in results:

    sol = result[0]
    SOI_exit_index = None

    if len(sol[2][1]) > 0:
        t_exits = sol[2][1]
        last_exit = t_exits[-1]
        SOI_exit_index = np.argmin(np.abs(sol[0] - last_exit))

    if len(sol[2][0]) > 0:
        collisions += 1
    

    if (accept_final_condition(sol[1][:3, :], sol[1][3:, :], mu, filters, sol=sol, SOI_exit_index=SOI_exit_index, rev_n=rev_n) or BYPASS):
        forward_results.append(result)



# Integrate retrograde trajectories
if RETROGRADE:
    transition_index = 0
    x0_retro = [result[0][1][:, 0] for result in forward_results]
    T_0_retro = sol[0][0]

    with get_context("fork").Pool() as pool:
        retro_results = pool.starmap(integrate_one, ((x0, T_0_retro, T_retro, dt_retro) for x0 in x0_retro))
else:
    retro_results = [None] * len(forward_results)
    transition_index = 0



# Fill the lists with the accepted trajectories
for forward, retro in zip(forward_results, retro_results):

    sol_f, traj_f, bar_f, moon_f, earth_f, e_f, m_f, Cj_f = forward

    if retro is None:
        times.append(sol_f[0])
        SV.append(traj_f)
        SV_inertial.append(bar_f)
        moon_inertial.append(bar_f - moon_f)
        earth_inertial.append(bar_f - earth_f)
        earth_energy.append(e_f)
        moon_energy.append(m_f)
        Cj.append(Cj_f)
        continue
    
    sol_r, traj_r, bar_r, moon_r, earth_r, e_r, m_r, Cj_r = retro

    transition_index = len(sol_r[0]) - 1

    times.append(np.concatenate((sol_r[0][::-1], sol_f[0])))
    SV.append(np.concatenate((traj_r[:, ::-1], traj_f), axis=1))
    SV_inertial.append(np.concatenate((bar_r[:, ::-1], bar_f), axis=1))
    moon_inertial.append(np.concatenate((bar_r[:, ::-1] - moon_r[:, ::-1], bar_f - moon_f), axis=1))
    earth_inertial.append(np.concatenate((bar_r[:, ::-1] - earth_r[:, ::-1], bar_f - earth_f), axis=1))
    moon_energy.append(np.concatenate((m_r[::-1], m_f), axis=0))
    earth_energy.append(np.concatenate((e_r[::-1], e_f), axis=0))
    Cj.append(np.concatenate((Cj_r[::-1], Cj_f), axis=0))



# Compute Inertial Trajectories
T = np.linspace(T_min, T_max, 10000)
Earth = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], 1, T)
Moon = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], 1, T)



# Print results
total = len(initial_conditions)
print("\n=== Simulation Results ===")
print(f"Total computed trajectories: {total}")
print(f"Total accepted trajectories: {len(SV)}")
print(f"Collisions:                       {collisions} ({100 * collisions / total:.1f} %)")


elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.2f} s")


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
        ax2.scatter(traj[0, transition_index], traj[1, transition_index], s=5, zorder=10)

    ax2.scatter(Earth[0, 0], Earth[1, 0], s=50, color='b', label='Earth', zorder=0)
    ax2.scatter(Moon[0, 0], Moon[1, 0], s=50, color='darkred', label='Moon', zorder=0) 



    ax3.set_title("Moon Inertial Frame")  

    ax3.scatter(0, 0, s=50, color='darkred', label='Moon', zorder=0) 
    
    for traj in moon_inertial:

        ax3.plot(traj[0, :], traj[1, :], alpha=0.8)
        ax3.scatter(traj[0, transition_index], traj[1, transition_index], s=5, zorder=10)


    
    for time, Cj_traj, e_energy, m_energy in zip(times, Cj, earth_energy, moon_energy):

        ax4.plot(time, Cj_traj - Cj_traj[0], alpha=0.8)
        ax5.plot(time, e_energy, alpha=0.8)
        ax6.plot(time, m_energy, alpha=0.8)

    moon_earth_energy = two_body_energy([1 - mu, 0, 0], [0, 0, 0], mu, body=1, frame='rotating')

    ax4.axhline(0, color='k', linestyle='--', alpha=0.8)
    ax5.axhline(0, color='k', linestyle='--', alpha=0.8)
    ax5.axvline(0, color='k', linestyle='--', alpha=0.8)
    ax5.axhline(moon_earth_energy, color='darkred', linestyle='--', alpha=0.8, label='Moon Energy')
    ax6.axhline(0, color='k', linestyle='--', alpha=0.8)
    ax6.axvline(0, color='k', linestyle='--', alpha=0.8)

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
    ax5.legend()

    ax6.set_title("Moon Specific Energy")
    ax6.set_xlabel("Time [TU]")
    ax6.set_ylabel(r"$\varepsilon_M$")

    plt.show()